import math
from typing import Union, List, Optional
import torch
import torch.nn.functional as F
from einops import rearrange
from packaging import version
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import PreTrainedModel, PretrainedConfig

try:
    import flash_attn
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False

# --- Config ---
class ReveConfig(PretrainedConfig):
    model_type = "reve"

    def __init__(
        self,
        embed_dim=512,
        depth=12,
        heads=8,
        head_dim=64,
        mlp_dim_ratio=4.0,
        use_geglu=True,
        freqs=4,
        noise_ratio=0.0025,
        patch_size=200,
        patch_overlap=0, # Default to 0 for ptft integration
        in_channels=19,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.depth = depth
        self.heads = heads
        self.head_dim = head_dim
        self.mlp_dim_ratio = mlp_dim_ratio
        self.use_geglu = use_geglu
        self.freqs = freqs
        self.noise_ratio = noise_ratio
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.in_channels = in_channels

# --- Layers ---
class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return F.gelu(gates) * x

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, geglu: bool):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, hidden_dim * 2 if geglu else hidden_dim, bias=False),
            GEGLU() if geglu else nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# --- Attention ---
class ClassicalAttention(nn.Module):
    def __init__(self, heads: int, use_sdpa: bool = True):
        super().__init__()
        self.use_sdpa = use_sdpa
        self.heads = heads

    def forward(self, qkv: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))

        if self.use_sdpa:
            # Check PyTorch version and mask support
            # For SDPA, attn_mask shape:
            # (L, S) or (N*H, L, S)
            # If key_padding_mask is (B, S), we need to expand/reshape?
            # F.scaled_dot_product_attention supports 'attn_mask' but it's usually additive or boolean mask.
            # However, it doesn't directly support key_padding_mask argument like MultiheadAttention.
            # We must convert key_padding_mask to attn_mask if provided.
            
            attn_mask = None
            if key_padding_mask is not None:
                # key_padding_mask: (B, S), True=Ignore
                # We need (B, 1, 1, S) or similar to broadcast?
                # Actually SDPA expects (Batch, Heads, Query, Key) or broadcastable.
                # Here Query=Key=S.
                # Mask should be (B, 1, S, S) or (B, 1, 1, S) if only masking keys?
                # Wait, causal mask is (S, S). Padding mask is usually just on keys.
                # (B, 1, 1, S) works if we want to mask keys for all queries.
                # 0 for keep, -inf for mask.
                
                # Create additive mask
                mask_val = -torch.finfo(q.dtype).max # Very small number
                # key_padding_mask is Bool (True=Ignore)
                # We want 0 where False, -inf where True
                attn_mask = torch.zeros_like(key_padding_mask, dtype=q.dtype)
                attn_mask.masked_fill_(key_padding_mask, mask_val)
                # Expand to (B, 1, 1, S)
                attn_mask = attn_mask.view(q.size(0), 1, 1, -1)
            
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            _, _, scale = q.shape[-2], q.device, q.shape[-1] ** -0.5
            dots = torch.matmul(q, k.transpose(-1, -2)) * scale
            
            if key_padding_mask is not None:
                # key_padding_mask: (B, S)
                # dots: (B, H, S, S)
                # Expand mask to (B, 1, 1, S)
                mask = key_padding_mask.unsqueeze(1).unsqueeze(1) # (B, 1, 1, S)
                dots = dots.masked_fill(mask, -torch.finfo(dots.dtype).max)
                
            attn = nn.Softmax(dim=-1)(dots)
            out = torch.matmul(attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return out

class FlashAttention(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, qkv: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Flash Attention implementation provided uses flash_attn_varlen_qkvpacked_func
        # which requires cu_seqlens.
        # But here inputs are padded (B, N, D).
        # We should use flash_attn_qkvpacked_func (non-varlen) if padded?
        # OR we convert padding mask to cu_seqlens and use varlen.
        
        # If we just want to use the padding mask, flash_attn 2.x supports it directly?
        # No, it usually requires unpadding.
        
        # Simplified: If mask is provided, use varlen implementation by unpadding.
        # If no mask, use batch implementation.
        
        batch_size, seq_len = qkv.shape[:2]
        
        if key_padding_mask is not None:
             # Unpad
             # key_padding_mask: (B, N) -> True if padding
             # We want valid tokens.
             valid_mask = ~key_padding_mask
             
             # Calculate cu_seqlens
             # We need to know how many valid tokens per batch element
             seqlens = valid_mask.sum(dim=1, dtype=torch.int32)
             cu_seqlens = torch.cat([torch.tensor([0], device=qkv.device, dtype=torch.int32), seqlens.cumsum(0, dtype=torch.int32)])
             max_seqlen = seqlens.max().item()
             
             # Flatten valid tokens
             # qkv: (B, N, 3, H, D)
             # We select valid tokens
             # We need to make sure qkv layout matches what we select
             # rearrange to (B, N, ...)
             qkv_flat = qkv[valid_mask] # (TotalValid, 3, H, D)
             
             out_flat = flash_attn.flash_attn_varlen_qkvpacked_func(
                 qkv_flat,
                 cu_seqlens,
                 max_seqlen,
                 0.0,
                 causal=False
             )
             
             # Repad output
             # out_flat: (TotalValid, H, D)
             # We need to scatter back to (B, N, H, D)
             # We can init zeros
             out = torch.zeros(batch_size, seq_len, self.num_heads * qkv.shape[-1], device=qkv.device, dtype=qkv.dtype)
             
             # But out shape from flash_attn is (TotalValid, H, D)
             # We need to combine H and D -> H*D
             out_flat = rearrange(out_flat, "t h d -> t (h d)")
             out[valid_mask] = out_flat
             
             return out
             
        else:
            # Original implementation (assumes no padding or user handles it)
            # But the original implementation used varlen with full length?
            # "cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, seq_len, ..."
            # This treats it as packed but with fixed length.
            # We can keep using it if no mask.
            
            qkv = rearrange(qkv, "b n (three h d) -> (b n) three h d", three=3, h=self.num_heads)
            cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=qkv.device)
            
            out = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens,
                seq_len,  # max seq len
                0.0,
                causal=False,
            )
    
            out = rearrange(out, "(b n) h d -> b n (h d)", b=batch_size)
            return out

class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, head_dim: int = 64, use_flash: bool = True):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.attend = FlashAttention(self.heads) if use_flash else ClassicalAttention(self.heads, use_sdpa=True)

    def forward(self, x, mask=None):
        x = self.norm(x)
        qkv = self.to_qkv(x)
        out = self.attend(qkv, key_padding_mask=mask)
        return self.to_out(out)

class TransformerBackbone(nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, geglu):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Attention(self.dim, heads=heads, head_dim=head_dim, use_flash=FLASH_AVAILABLE),
                    FeedForward(self.dim, mlp_dim, geglu),
                ])
            )

    def forward(self, x, return_out_layers=False, mask=None) -> Union[torch.Tensor, List[torch.Tensor]]:
        out_layers = [x] if return_out_layers else None
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
            if return_out_layers:
                out_layers.append(x)
        return out_layers if return_out_layers else x

# --- Positional Embeddings ---
class FourierEmb4D(nn.Module):
    def __init__(self, dimension: int, freqs: int, increment_time=0.1, margin: float = 0.4):
        super().__init__()
        self.dimension = dimension
        self.freqs = freqs
        self.increment_time = increment_time
        self.margin = margin

    def forward(self, positions_):
        positions = positions_.clone()
        positions[:, :, -1] *= self.increment_time
        *U, _ = positions.shape
        freqs_w = torch.arange(self.freqs).to(positions)
        freqs_z = freqs_w[:, None]
        freqs_y = freqs_z[:, None]
        freqs_x = freqs_y[:, None]
        width = 1 + 2 * self.margin
        positions = positions + self.margin
        p_x = 2 * math.pi * freqs_x / width
        p_y = 2 * math.pi * freqs_y / width
        p_z = 2 * math.pi * freqs_z / width
        p_w = 2 * math.pi * freqs_w / width
        positions = positions[..., None, None, None, None, :]
        loc = (positions[..., 0] * p_x + positions[..., 1] * p_y + positions[..., 2] * p_z + positions[..., 3] * p_w).view(*U, -1)
        if self.dimension != 512:
            _, _, hd = loc.shape
            diff = hd - self.dimension // 2
            loc = loc[:, :, :-diff]
        emb = torch.cat([torch.cos(loc), torch.sin(loc)], dim=-1)
        return emb

    @classmethod
    def add_time_patch(cls, pos, num_patches):
        B, C, _ = pos.shape
        pos_repeated = pos.unsqueeze(2).repeat(1, 1, num_patches, 1)
        time_values = torch.arange(0, num_patches, 1, device=pos.device).float()
        time_values = time_values.view(1, 1, num_patches, 1).expand(B, C, num_patches, 1)
        pos_with_time = torch.cat((pos_repeated, time_values), dim=-1)
        pos_with_time = pos_with_time.view(B, C * num_patches, 4)
        return pos_with_time

def patch_embedding_layer(embed_dim, patch_size):
    return nn.Sequential(nn.Linear(patch_size, embed_dim))

def mlp_pos_embedding_layer(embed_dim):
    return nn.Sequential(nn.Linear(4, embed_dim, bias=False), nn.GELU(), nn.LayerNorm(embed_dim))

# --- Standard 10-20 Coordinates ---
STANDARD_1020 = {
    'FP1': [-0.309, 0.951, 0.0],
    'FP2': [0.309, 0.951, 0.0],
    'F7': [-0.809, 0.588, -0.0],
    'F3': [-0.54, 0.67, 0.5],
    'FZ': [0.0, 0.71, 0.71],
    'F4': [0.54, 0.67, 0.5],
    'F8': [0.809, 0.588, -0.0],
    'T3': [-0.951, 0.0, -0.309],
    'C3': [-0.71, 0.0, 0.71],
    'CZ': [0.0, 0.0, 1.0],
    'C4': [0.71, 0.0, 0.71],
    'T4': [0.951, 0.0, -0.309],
    'T5': [-0.809, -0.588, -0.0],
    'P3': [-0.54, -0.67, 0.5],
    'PZ': [0.0, -0.71, 0.71],
    'P4': [0.54, -0.67, 0.5],
    'T6': [0.809, -0.588, -0.0],
    'O1': [-0.309, -0.951, 0.0],
    'O2': [0.309, -0.951, 0.0],
    # Mappings for alternative names
    'T7': [-0.951, 0.0, -0.309], # Same as T3
    'T8': [0.951, 0.0, -0.309], # Same as T4
    'P7': [-0.809, -0.588, -0.0], # Same as T5
    'P8': [0.809, -0.588, -0.0], # Same as T6
}

# --- Reve Backbone Wrapper for PTFT ---
class ReveBackbone(nn.Module):
    def __init__(self, in_dim=200, d_model=512, seq_len=60, n_layer=12, nhead=8, **kwargs):
        super().__init__()
        # Map ptft params to ReveConfig
        config = ReveConfig(
            embed_dim=d_model,
            depth=n_layer,
            heads=nhead,
            patch_size=in_dim,
            # Additional params can be passed via kwargs
            **kwargs
        )
        self.config = config
        
        self.transformer = TransformerBackbone(
            dim=config.embed_dim,
            depth=config.depth,
            heads=config.heads,
            head_dim=config.head_dim,
            mlp_dim=int(config.embed_dim * config.mlp_dim_ratio),
            geglu=config.use_geglu,
        )

        self.to_patch_embedding = patch_embedding_layer(config.embed_dim, config.patch_size)
        self.fourier4d = FourierEmb4D(config.embed_dim, freqs=config.freqs)
        self.mlp4d = mlp_pos_embedding_layer(config.embed_dim)
        self.ln = nn.LayerNorm(config.embed_dim)
        
        # Mask Token for MAE
        self.mask_encoding = nn.Parameter(torch.zeros(config.embed_dim), requires_grad=True)

        # Pre-compute positions for standard channels
        # Order must match TUEG dataset: 
        # ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ']
        self.channel_order = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
                              'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ']
        
        pos_list = []
        for ch in self.channel_order:
            if ch in STANDARD_1020:
                pos_list.append(STANDARD_1020[ch])
            else:
                # Fallback to origin if not found
                pos_list.append([0.0, 0.0, 0.0])
                
        self.register_buffer('standard_pos', torch.tensor(pos_list, dtype=torch.float32))

    def forward(self, x, mask=None, channel_mask=None, time_mask=None):
        # x: (B, C, N, P)
        B, C, N, P = x.shape
        
        # 1. Patch Embedding
        # Apply mask if present (MAE)
        if mask is not None:
            # mask: (B, C, N) -> 1 is masked, 0 is visible (usually)
            # OR 0 is masked, 1 is visible?
            # In ptft wrapper: mask = 1 means masked (Bernoulli 0.5)
            # We want to replace masked patches with mask_encoding
            
            # Embed patches first
            x_emb = self.to_patch_embedding(x) # (B, C, N, E)
            
            # Expand mask to (B, C, N, E)
            mask_expanded = mask.unsqueeze(-1).expand_as(x_emb)
            
            # Apply mask replacement
            # Note: mask values are 0 or 1. If 1 is masked:
            x_emb = x_emb * (1 - mask.unsqueeze(-1)) + self.mask_encoding * mask.unsqueeze(-1)
        else:
            x_emb = self.to_patch_embedding(x)

        # Apply channel_mask and time_mask to zero out padded regions in x_emb
        # channel_mask: (B, C) - True if padded
        # time_mask: (B, N) - True if padded
        if channel_mask is not None:
             x_emb = x_emb.masked_fill(channel_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
        if time_mask is not None:
             x_emb = x_emb.masked_fill(time_mask.unsqueeze(1).unsqueeze(-1), 0.0)
            
        # 2. Positional Embedding
        # standard_pos: (C, 3) -> (B, C, 3)
        # Note: In Joint Dataset, C can be variable and padded.
        # But standard_pos is fixed (19). 
        # If x has more channels than 19, we need to handle it.
        # However, ptft usually assumes a fixed superset or maps to fixed channels.
        # If x has FEWER channels (padded), we just use the first C positions.
        # BUT wait, the channel order matters.
        # In JointDataset, we collate tensors of shape (C_i, N_i, P).
        # The channels are usually mapped to a standard set in the dataset loader.
        # TUEGDataset maps to 19 standard channels.
        # If other datasets map to subsets, they might be padded.
        # We assume the channels in x correspond to self.channel_order indices.
        
        # If x.shape[1] > len(self.standard_pos), we might have an issue unless we expand standard_pos.
        # For now, we assume x.shape[1] <= 19 or we slice standard_pos.
        # Actually, let's use the actual C from input.
        
        num_known_channels = self.standard_pos.shape[0]
        if C <= num_known_channels:
             current_pos = self.standard_pos[:C].unsqueeze(0).expand(B, -1, -1)
        else:
             # Fallback: repeat last position or pad with zeros?
             # Or just panic. Let's pad with zeros.
             padding = torch.zeros(C - num_known_channels, 3, device=x.device)
             current_pos = torch.cat([self.standard_pos, padding], dim=0).unsqueeze(0).expand(B, -1, -1)
        
        # Add time dimension to pos: (B, C*N, 4)
        pos_4d = FourierEmb4D.add_time_patch(current_pos, N)
        
        # Compute pos embedding
        pos_embed = self.ln(self.fourier4d(pos_4d) + self.mlp4d(pos_4d)) # (B, C*N, E)
        
        # 3. Prepare for Transformer
        # Flatten C and N dimensions
        x_flat = rearrange(x_emb, "b c n e -> b (c n) e")
        
        # Add pos embedding
        x_flat = x_flat + pos_embed
        
        # 4. Transformer Forward
        # Create attention mask for padded regions
        # channel_mask: (B, C) -> True if padded (ignore)
        # time_mask: (B, N) -> True if padded (ignore)
        # We need a combined mask of shape (B, C*N)
        
        key_padding_mask = None
        if channel_mask is not None or time_mask is not None:
             # Create full mask (B, C, N)
             full_mask = torch.zeros(B, C, N, dtype=torch.bool, device=x.device)
             
             if channel_mask is not None:
                 # channel_mask is (B, C). If True, that channel is padded for all time steps.
                 full_mask = full_mask | channel_mask.unsqueeze(-1)
                 
             if time_mask is not None:
                 # time_mask is (B, N). If True, that time step is padded for all channels.
                 full_mask = full_mask | time_mask.unsqueeze(1)
                 
             # Flatten to (B, C*N)
             key_padding_mask = rearrange(full_mask, "b c n -> b (c n)")
             
             # Note: Transformer expects key_padding_mask where True means IGNORE.
             # This matches our convention from collate_joint.
        
        # We need to pass key_padding_mask to TransformerBackbone -> Attention
        # But our TransformerBackbone/Attention implementation doesn't seem to take a mask argument in forward?
        # Let's check TransformerBackbone.forward signature.
        # It takes 'x' and 'return_out_layers'.
        # We need to update TransformerBackbone to accept mask.
        
        # For now, we proceed without mask if underlying transformer doesn't support it,
        # BUT this is critical for variable length.
        # The provided modeling_reve.py shows Attention class takes 'x'.
        # Flash Attention supports 'cu_seqlens' for variable length, but here we have padding mask.
        # If using Flash Attention with padding, we usually need to unpad or pass mask.
        # The FlashAttention implementation provided (flash_attn_varlen_qkvpacked_func) is for variable length (packed).
        # But here we are using padded tensors.
        # If we use ClassicalAttention, we can pass mask.
        
        # Assuming we need to modify TransformerBackbone to accept mask.
        # Let's assume we can modify it.
        
        x_out = self.transformer(x_flat, mask=key_padding_mask)
        
        # 5. Reshape back to (B, C, N, E)
        x_out = rearrange(x_out, "b (c n) e -> b c n e", c=C, n=N)
        
        return x_out
