import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.config_mamba import MambaConfig
from .modules.mixer_seq_simple import MixerModel
from torch.nn.modules import TransformerEncoderLayer, TransformerEncoder
from einops import rearrange


class EEGMambaBackbone(nn.Module):
    def __init__(self, in_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_dim, d_model, seq_len)
        config = MambaConfig()
        config.d_model = d_model
        config.n_layer = n_layer
        config.d_intermediate = 0
        config.rms_norm = False
        config.fused_add_norm = False
        config.ssm_cfg = {
            # Mamba1 is numerically more stable than Mamba2 in the current runtime stack.
            "layer": "Mamba1",
            "d_state": 64,
        }
        self.encoder = MixerModel(
            d_model=config.d_model,
            n_layer=config.n_layer,
            d_intermediate=config.d_intermediate,
            ssm_cfg=config.ssm_cfg,
            attn_layer_idx=config.attn_layer_idx,
            attn_cfg=config.attn_cfg,
            rms_norm=config.rms_norm,
            initializer_cfg=None,
            fused_add_norm=config.fused_add_norm,
            residual_in_fp32=config.residual_in_fp32,
        )
        self.apply(_weights_init)

    def forward(self, x, mask=None, channel_mask=None, time_mask=None):
        bz, ch_num, seq_len, patch_size = x.shape
        if channel_mask is not None:
            x = x.masked_fill(channel_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
        if time_mask is not None:
            x = x.masked_fill(time_mask.unsqueeze(1).unsqueeze(-1), 0.0)
            
        hidden_states = self.patch_embedding(x, mask=mask)
        hidden_states = rearrange(hidden_states, 'b c l d -> b (c l) d')
        hidden_states = self.encoder(hidden_states)
        
        hidden_states = rearrange(hidden_states, 'b (c l) d -> b c l d', l=seq_len)
        return hidden_states


class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        # self.norm = nn.InstanceNorm2d(200)
        self.positional_encoding = nn.Sequential(
            # nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4),
            #           groups=d_model, bias=False),
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3),
                      groups=d_model, bias=False),
            # nn.GroupNorm(40, d_model),
            # nn.GELU(),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        # self.mask_encoding = nn.Parameter(torch.randn(in_dim), requires_grad=True)

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24), bias=False),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            # nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            # nn.GroupNorm(5, 25),
            # nn.GELU(),
            #
            # nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            # nn.GroupNorm(5, 25),
            # nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(101, d_model, bias=False),
            nn.Dropout(0.1),
        )


    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        if mask == None:
            mask_x = x
        else:
            mask_x = x.clone()
            # Handle mask broadcast: mask is (B, C, N)
            # mask_x is (B, C, N, P)
            # We need to broadcast mask to mask_x shape
            # mask.unsqueeze(-1) -> (B, C, N, 1)
            mask_x[mask.unsqueeze(-1).expand_as(mask_x) == 1] = self.mask_encoding.repeat(mask_x[mask.unsqueeze(-1).expand_as(mask_x) == 1].numel() // self.mask_encoding.numel())

        mask_x = rearrange(mask_x, 'b c l d -> b d c l')
        # norm_x = self.norm(mask_x)
        # norm_x = mask_x
        time_x = rearrange(mask_x, 'b d c l -> b (c l) d').unsqueeze(1)

        time_emb = self.proj_in(time_x)
        time_emb = time_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)

        freq_x = rearrange(mask_x, 'b d c l -> b c l d')
        spectral = torch.fft.rfft(freq_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral)
        spectral_emb = self.spectral_proj(spectral)
        patch_emb = time_emb + spectral_emb

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)

        patch_emb = patch_emb + positional_embedding

        return patch_emb



def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
