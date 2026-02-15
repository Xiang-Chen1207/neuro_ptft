import torch
import torch.nn as nn
import torch.nn.functional as F

from .criss_cross_transformer import TransformerEncoderLayer, TransformerEncoder

class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, seq_len):
        super().__init__()
        # print(f"DEBUG: PatchEmbedding init: in_dim={in_dim}, out_dim={out_dim}, d_model={d_model}, seq_len={seq_len}")
        self.d_model = d_model
        
        # Determine patch size and FFT size
        # in_dim is expected to be input_size (e.g. 12000) or patch_size?
        # Based on forward: x shape is (bz, ch_num, patch_num, patch_size)
        # So in_dim passed here is likely irrelevant or confusingly named if it's not patch_size.
        # However, backbone init takes in_dim=200, which is patch_size.
        
        # Hardcoded params exposed
        self.patch_size = in_dim 
        self.freq_bins = self.patch_size // 2 + 1 # e.g. 200 -> 101
        self.conv_channels = 25 # Could be config param
        self.conv_out_len = (self.patch_size + 2 * 24 - 49) // 25 + 1
        self.time_feat_dim = self.conv_channels * self.conv_out_len
        
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(19, 7), stride=(1, 1), padding=(9, 3),
                      groups=d_model),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        # self.mask_encoding = nn.Parameter(torch.randn(in_dim), requires_grad=True)

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.conv_channels, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, self.conv_channels),
            nn.GELU(),

            nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, self.conv_channels),
            nn.GELU(),

            nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, self.conv_channels),
            nn.GELU(),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_feat_dim, d_model),
            nn.Dropout(0.1),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(self.freq_bins, d_model),
            nn.Dropout(0.1),
            # nn.LayerNorm(d_model, eps=1e-5),
        )

    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        if mask == None:
            mask_x = x
        else:
            mask_x = x.clone()
            # Explicit broadcast for safety and clarity
            # mask: (B, C, N) -> (B, C, N, 1)
            # mask_encoding: (P,) -> (1, 1, 1, P)
            # This line works due to numpy-style advanced indexing, but explicit is better if dimensions mismatch
            # mask_x[mask == 1] = self.mask_encoding
            
            # Safer implementation:
            mask_bool = (mask == 1).unsqueeze(-1) # (B, C, N, 1)
            mask_x = torch.where(mask_bool, self.mask_encoding.view(1, 1, 1, -1), mask_x)

        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        time_feat = self.proj_in(mask_x)
        time_feat = time_feat.permute(0, 2, 1, 3).contiguous().view(bz, ch_num * patch_num, self.time_feat_dim)
        # print(f"DEBUG: forward: time_feat shape={time_feat.shape}, self.d_model={self.d_model}")
        patch_emb = self.time_proj(time_feat).contiguous().view(bz, ch_num, patch_num, self.d_model)

        mask_x = mask_x.contiguous().view(bz*ch_num*patch_num, patch_size)
        spectral = torch.fft.rfft(mask_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, self.freq_bins)
        spectral_emb = self.spectral_proj(spectral)
        patch_emb = patch_emb + spectral_emb

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)

        patch_emb = patch_emb + positional_embedding

        return patch_emb

class CBraModBackbone(nn.Module):
    def __init__(self, in_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_dim, d_model, d_model, seq_len)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, norm_first=True,
            activation=F.gelu
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=n_layer, enable_nested_tensor=False)
        self.apply(self._weights_init)

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, mask=None, channel_mask=None, time_mask=None):
        patch_emb = self.patch_embedding(x, mask)
        feats = self.encoder(patch_emb, channel_mask=channel_mask, time_mask=time_mask)
        return feats
