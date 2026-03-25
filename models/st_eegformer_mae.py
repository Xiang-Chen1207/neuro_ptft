
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        # Note: drop_path is not implemented here for simplicity
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, key_padding_mask=None):
        # MultiheadAttention returns (attn_output, attn_output_weights)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask=key_padding_mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbed(nn.Module):
    """ EEG to Patch Embedding
    """
    def __init__(self, num_channels=20, time_len=12000, patch_size=200, embed_dim=200):
        super().__init__()
        self.num_channels = num_channels
        self.time_len = time_len
        self.patch_size = patch_size
        self.num_patches = time_len // patch_size
        
        self.proj = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        # x: (B, C, T) or (B, C, N, P) if already patchified
        # The dataloader returns (B, C, N, P)
        
        if x.ndim == 4:
            B, C, N, P = x.shape
        else:
            B, C, T = x.shape
            N = T // self.patch_size
            x = x.view(B, C, N, self.patch_size)
        
        # Linear Projection
        # (B, C, N, D)
        x = self.proj(x)
        return x

class ST_EEGFormer_MAE(nn.Module):
    def __init__(self, 
                 in_dim=200, # patch_size
                 d_model=512, # encoder_dim
                 depth=8, 
                 nhead=8, 
                 mlp_ratio=4., 
                 decoder_depth=4,
                 decoder_embed_dim=384,
                 decoder_num_heads=16,
                 decoder_mlp_ratio=4.,
                 num_channels=20, # Max channels, will be sliced if input is smaller? Or fixed?
                 # Assuming fixed max channels or handling variable channels via masking/embedding?
                 # For simplicity, let's assume max channels or use broadcasting for SPE if channels vary?
                 # SPE is (1, C, 1, D). If input has c < C, we slice SPE?
                 # Ideally, we should use channel_id to index SPE.
                 # But standard implementation assumes fixed channels.
                 # Let's assume input channels <= num_channels.
                 seq_len=60, # num_patches
                 norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.patch_size = in_dim
        self.num_patches = seq_len
        self.num_channels = num_channels # Max supported channels
        
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(num_channels, seq_len*in_dim, in_dim, d_model)
        
        # Positional Embeddings
        # Temporal (Sine-Cosine) - Fixed
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, 1, seq_len, d_model), requires_grad=False)
        pos_embed = get_1d_sincos_pos_embed_from_grid(d_model, np.arange(seq_len, dtype=np.float32))
        self.temporal_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0).unsqueeze(0))
        
        # Spatial (Learned)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_channels, 1, d_model))
        nn.init.normal_(self.spatial_pos_embed, std=0.02)
        
        # Encoder Blocks
        self.blocks = nn.ModuleList([
            Block(d_model, nhead, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(d_model)
        
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(d_model, decoder_embed_dim, bias=True)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Decoder Positional Embeddings (Need to match Decoder Dim)
        self.decoder_temporal_pos_embed = nn.Parameter(torch.zeros(1, 1, seq_len, decoder_embed_dim), requires_grad=False)
        dec_pos_embed = get_1d_sincos_pos_embed_from_grid(decoder_embed_dim, np.arange(seq_len, dtype=np.float32))
        self.decoder_temporal_pos_embed.data.copy_(torch.from_numpy(dec_pos_embed).float().unsqueeze(0).unsqueeze(0))
        
        self.decoder_spatial_pos_embed = nn.Parameter(torch.zeros(1, num_channels, 1, decoder_embed_dim))
        nn.init.normal_(self.decoder_spatial_pos_embed, std=0.02)
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, decoder_mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_dim, bias=True) # reconstruction
        # --------------------------------------------------------------------------

    def _get_pos_embed(self, pos_embed, seq_len, d_model):
        if pos_embed.shape[2] >= seq_len:
            return pos_embed[:, :, :seq_len, :]
        else:
            # Dynamic generation for longer sequences
            # Assuming get_1d_sincos_pos_embed_from_grid is available in global scope
            new_pos_embed = get_1d_sincos_pos_embed_from_grid(d_model, np.arange(seq_len, dtype=np.float32))
            return torch.from_numpy(new_pos_embed).float().unsqueeze(0).unsqueeze(0).to(pos_embed.device)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x, mask_ratio, channel_mask=None, time_mask=None):
        # x: (B, C, N, P)
        
        # embed patches
        x = self.patch_embed(x) # (B, C, N, D)
        B, C, N, D = x.shape
        
        # add pos embed
        # Slice spatial embed to match input channels
        if self.spatial_pos_embed.shape[1] < C:
             # Fallback: repeat or error? 
             # For now, let's assume max channels is sufficient or we need to expand.
             # Expanding learned embed is tricky. Let's slice if possible.
             # If C > max, we slice max. But usually C <= max.
             spe = self.spatial_pos_embed[:, :C, :, :]
        else:
             spe = self.spatial_pos_embed[:, :C, :, :]

        tpe = self._get_pos_embed(self.temporal_pos_embed, N, D)
        x = x + tpe + spe
        
        # flatten (B, C*N, D)
        x = x.view(B, C*N, D)
        
        # Construct padding mask (B, C*N)
        padding_mask = None
        if channel_mask is not None or time_mask is not None:
            if channel_mask is None:
                channel_mask = torch.zeros(B, C, dtype=torch.bool, device=x.device)
            if time_mask is None:
                time_mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
            
            # (B, C, 1) | (B, 1, N) -> (B, C, N)
            padding_mask_2d = channel_mask.unsqueeze(2) | time_mask.unsqueeze(1)
            padding_mask = padding_mask_2d.view(B, C*N)
        
        # masking: generate random mask
        x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        
        # Gather padding mask for encoder
        enc_padding_mask = None
        if padding_mask is not None:
            # (B, len_keep)
            enc_padding_mask = torch.gather(padding_mask, dim=1, index=ids_keep)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, key_padding_mask=enc_padding_mask)
        x = self.norm(x)

        return x, mask, ids_restore, padding_mask

    def forward_decoder(self, x, ids_restore, B, C, N, padding_mask=None):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = x_

        # add pos embed
        # reshape to (B, C, N, D) to add spatial/temporal embed correctly
        x = x.view(B, C, N, -1)
        
        decoder_dim = x.shape[-1]
        spe = self.decoder_spatial_pos_embed[:, :C, :, :]
        tpe = self._get_pos_embed(self.decoder_temporal_pos_embed, N, decoder_dim)
        
        x = x + tpe + spe
        x = x.view(B, C*N, -1)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, key_padding_mask=padding_mask)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward(self, x, mask_ratio=0.75, channel_mask=None, time_mask=None):
        # x: (B, C, N, P)
        B, C, N, P = x.shape
        
        latent, mask, ids_restore, padding_mask = self.forward_encoder(x, mask_ratio, channel_mask, time_mask)
        pred = self.forward_decoder(latent, ids_restore, B, C, N, padding_mask)  # [B, C*N, P]
        
        # Reshape pred to (B, C, N, P)
        pred = pred.view(B, C, N, P)
        
        # Reshape mask to (B, C, N)
        mask = mask.view(B, C, N)
        
        return pred, mask
