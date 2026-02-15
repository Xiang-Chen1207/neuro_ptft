
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

class STEEGFormer(nn.Module):
    def __init__(self, in_dim=200, d_model=1024, seq_len=60, n_layer=24, nhead=16, mlp_ratio=4, dropout=0.0, 
                 decoder_depth=8, decoder_embed_dim=512, decoder_num_heads=16, decoder_mlp_ratio=4,
                 task_type='pretraining'):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.task_type = task_type
        
        # 3.1 & 6.1 Patch Embedding
        # Linear projection of flattened patches
        self.patch_embed = nn.Linear(in_dim, d_model)
        
        # 3.2 Positional Encoding
        # Temporal: Sine-Cosine
        self.register_buffer('temporal_pos_embed', self._get_sinusoidal_encoding(seq_len, d_model))
        # Spatial: Learned (Assuming max 128 channels)
        self.spatial_pos_embed = nn.Embedding(128, d_model)
        
        # 3.4 Encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, mlp_ratio, dropout)
            for _ in range(n_layer)
        ])
        self.norm = nn.LayerNorm(d_model)
        
        # 3.5 Decoder (Only for pretraining)
        if task_type == 'pretraining':
            self.decoder_embed = nn.Linear(d_model, decoder_embed_dim, bias=True)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            nn.init.normal_(self.mask_token, std=0.02)
            
            self.decoder_pos_embed = self._get_sinusoidal_encoding(seq_len, decoder_embed_dim)
            self.decoder_spatial_pos_embed = nn.Embedding(128, decoder_embed_dim)
            
            self.decoder_blocks = nn.ModuleList([
                TransformerBlock(decoder_embed_dim, decoder_num_heads, decoder_mlp_ratio, dropout)
                for _ in range(decoder_depth)
            ])
            self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
            self.decoder_pred = nn.Linear(decoder_embed_dim, in_dim, bias=True)
        else:
            self.decoder_blocks = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def _get_sinusoidal_encoding(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x, mask=None, channel_mask=None, time_mask=None):
        # x: (B, C, N, P)
        B, C, N, P = x.shape
        
        # 1. Embed
        x = self.patch_embed(x) # (B, C, N, D)
        
        # 2. Add Positional Embeddings (Encoder)
        tpe = self.temporal_pos_embed[:, :N, :].unsqueeze(1) # (1, 1, N, D)
        spe_idx = torch.arange(C, device=x.device).unsqueeze(0) # (1, C)
        spe = self.spatial_pos_embed(spe_idx).unsqueeze(2) # (1, C, 1, D)
        
        x = x + tpe + spe
        
        # Flatten to (B, L, D)
        x = rearrange(x, 'b c n d -> b (c n) d')
        
        # 3. Create Padding Mask (for Attention)
        # We assume channel_mask/time_mask indicates PADDING (True = Padded/Ignored)
        key_padding_mask = None
        if channel_mask is not None or time_mask is not None:
            c_mask = channel_mask.unsqueeze(2).expand(B, C, N) if channel_mask is not None else torch.zeros(B, C, N, dtype=torch.bool, device=x.device)
            t_mask = time_mask.unsqueeze(1).expand(B, C, N) if time_mask is not None else torch.zeros(B, C, N, dtype=torch.bool, device=x.device)
            full_mask = c_mask | t_mask # (B, C, N)
            key_padding_mask = rearrange(full_mask, 'b c n -> b (c n)') # (B, L)

        # 4. MAE Masking (If Pretraining)
        # Note: We implement "Simulated MAE" where we mask attention to dropped tokens
        # but keep tensor shape, to handle variable channels easily.
        # However, to match "Mask tokens not involved in encoder", we can replace them with a special token 
        # or use attention mask to block them. 
        # Here we use attention mask for "dropping" logic.
        
        mask_ratio = 0.75 if self.task_type == 'pretraining' else 0.0
        
        if self.task_type == 'pretraining':
            # Generate random mask for MAE (True = Masked/Dropped)
            # We want to mask 75% of *valid* tokens.
            # But for simplicity, we mask 75% of ALL tokens and combine with padding mask.
            mae_mask = torch.rand(B, C * N, device=x.device) < mask_ratio
            
            # Ensure padding tokens are NOT selected as "masked" for reconstruction (they are ignored)
            # But for Encoder input, they are "masked" (ignored).
            # Effectively: Encoder Visibility Mask = ~(Padding | MAE_Mask)
            
            if key_padding_mask is not None:
                # Ensure we don't calculate loss on padding
                mae_mask = mae_mask & (~key_padding_mask)
                combined_mask = key_padding_mask | mae_mask
            else:
                combined_mask = mae_mask
                
            # Run Encoder with combined mask
            enc_out = x
            for blk in self.blocks:
                enc_out = blk(enc_out, key_padding_mask=combined_mask)
            enc_out = self.norm(enc_out)
            
            # Decoder Logic
            # 1. Project to decoder dim
            dec_in = self.decoder_embed(enc_out)
            
            # 2. Add Mask Tokens (Replace masked positions with learnable token)
            # In standard MAE, we only feed visible to encoder, then cat mask tokens.
            # Here, since we fed everything (but masked attention), `enc_out` at masked positions 
            # is "garbage" (processed but attended only to visible). 
            # We should replace these positions with Mask Token now.
            
            mask_token_expand = self.mask_token.expand(B, C*N, -1)
            
            # Where mae_mask is True, use Mask Token. 
            # Note: We should NOT replace Padding tokens with Mask Tokens? 
            # Decoder needs to reconstruct "Masked" regions. It doesn't need to reconstruct "Padding".
            # So:
            # If (MAE_Masked AND NOT Padding) -> Mask Token
            # If (Visible) -> Enc Output
            # If (Padding) -> Don't care (keep Enc Output or Zero)
            
            # Logic: Use Mask Token where mae_mask is True.
            # (Assuming mae_mask was generated randomly over all, including padding).
            # We should probably zero out mae_mask on padding regions to be clean, 
            # but usually we just ignore loss on padding.
            
            # Apply Mask Token
            # (B, L, D_dec)
            dec_in = torch.where(mae_mask.unsqueeze(-1), mask_token_expand, dec_in)
            
            # 3. Add Decoder Positional Embeddings
            tpe_dec = self.decoder_pos_embed.to(x.device)[:, :N, :].unsqueeze(1) # (1, 1, N, D)
            spe_dec = self.decoder_spatial_pos_embed(spe_idx).unsqueeze(2)
            pos_dec = (tpe_dec + spe_dec) # (1, C, N, D)
            pos_dec = rearrange(pos_dec, '1 c n d -> 1 (c n) d')
            
            dec_in = dec_in + pos_dec
            
            # 4. Run Decoder
            # Decoder should see everything (Full Attention), except Padding.
            # So use key_padding_mask (which is just Padding).
            dec_out = dec_in
            for blk in self.decoder_blocks:
                dec_out = blk(dec_out, key_padding_mask=key_padding_mask)
            dec_out = self.decoder_norm(dec_out)
            
            # 5. Project to Output
            pred = self.decoder_pred(dec_out) # (B, L, P)
            
            # Reshape back
            pred = rearrange(pred, 'b (c n) p -> b c n p', c=C, n=N)
            
            # Return prediction and the mask used for loss
            # Reshape mask
            mae_mask = rearrange(mae_mask, 'b (c n) -> b c n', c=C, n=N)
            
            return pred, mae_mask
            
        else:
            # Downstream Task (Classification)
            # No masking, just Encoder
            for blk in self.blocks:
                x = blk(x, key_padding_mask=key_padding_mask)
            x = self.norm(x)
            
            # Reshape back
            x = rearrange(x, 'b (c n) d -> b c n d', c=C, n=N)
            return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )
        
    def forward(self, x, key_padding_mask=None):
        # x: (B, L, D)
        # key_padding_mask: (B, L) - True where we should IGNORE (padding/masked)
        
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x
