import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from collections import defaultdict


def generate_area_config(brain_regions: List[int]):
    region_to_channels = defaultdict(list)
    for channel_idx, region in enumerate(brain_regions):
        region_to_channels[int(region)].append(channel_idx)

    area_config = {}
    for region, channels in region_to_channels.items():
        area_config[f"region_{region}"] = {
            'channels': len(channels),
            'slice': slice(channels[0], channels[-1] + 1)
        }
    return area_config


class PatchEmbedding(nn.Module):
    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=(19, 7),
            stride=(1, 1),
            padding=(9, 3),
            groups=d_model,
        )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(d_model // 2 + 1, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        bz, ch_num, patch_num, patch_size = x.shape
        if mask is None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)

        flat_x = mask_x.contiguous().view(bz * ch_num * patch_num, patch_size)
        spectral = torch.fft.rfft(flat_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, flat_x.shape[1] // 2 + 1)
        patch_emb = patch_emb + self.spectral_proj(spectral)

        pos = self.positional_encoding(patch_emb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return patch_emb + pos


class TemEmbedEEGLayer(nn.Module):
    def __init__(self, d_model: int, kernel_sizes=None):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [1, 3, 5]
        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)
        dim_scales = [int(d_model / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, d_model - sum(dim_scales)]
        self.convs = nn.ModuleList([
            nn.Conv2d(d_model, dim_scale, kernel_size=(k, 1), stride=(1, 1), padding=((k - 1) // 2, 0))
            for k, dim_scale in zip(kernel_sizes, dim_scales)
        ])

    def forward(self, x: torch.Tensor):
        batch, chans, time, d_model = x.shape
        x = x.view(batch * chans, d_model, time, 1)
        fmaps = [conv(x) for conv in self.convs]
        x = torch.cat(fmaps, dim=1)
        return x.view(batch, chans, time, -1)


class BrainEmbedEEGLayer(nn.Module):
    def __init__(self, d_model=200, total_regions=8):
        super().__init__()
        kernel_sizes = [1, 3, 5]
        dim_scales = [d_model // (2 ** (i + 1)) for i in range(len(kernel_sizes) - 1)]
        dim_scales.append(d_model - sum(dim_scales))
        self.dim_out = d_model
        self.kernel_sizes = kernel_sizes
        self.region_blocks = nn.ModuleDict({
            f"region_{i}": nn.ModuleList([
                nn.Conv2d(d_model, dim_scale, kernel_size=(k, 1), padding=(0, 0))
                for k, dim_scale in zip(kernel_sizes, dim_scales)
            ])
            for i in range(total_regions)
        })

    def forward(self, x, area_config):
        batch, chans, t, f = x.shape
        out = torch.zeros((batch, chans, t, self.dim_out), device=x.device, dtype=x.dtype)
        for region_key, region_info in area_config.items():
            channel_slice = region_info['slice']
            n_electrodes = region_info['channels']
            x_region = x[:, channel_slice, :, :]
            x_trans = x_region.permute(0, 2, 1, 3).reshape(-1, n_electrodes, f)
            x_trans = x_trans.permute(0, 2, 1).unsqueeze(-1)
            if region_key not in self.region_blocks:
                continue
            blocks = self.region_blocks[region_key]
            fmap_outputs = []
            for conv, k in zip(blocks, self.kernel_sizes):
                pad = (k - 1) // 2
                if n_electrodes == 1:
                    x_padded = F.pad(x_trans, (0, 0, pad, pad), mode='constant', value=0)
                else:
                    x_padded = F.pad(x_trans, (0, 0, pad, pad), mode='circular')
                fmap_outputs.append(conv(x_padded))
            fmap_cat = torch.cat(fmap_outputs, dim=1)
            fmap_out = fmap_cat.squeeze(-1).permute(0, 2, 1).reshape(batch, t, n_electrodes, self.dim_out)
            out[:, channel_slice, :, :] = fmap_out.permute(0, 2, 1, 3)
        return out


class CSBrainEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 800, dropout: float = 0.1, area_config=None):
        super().__init__()
        self.inter_region_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.inter_window_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.global_fc = nn.Linear(d_model, d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.area_config = area_config or {}
        self.region_indices = {
            k: list(range(v['slice'].start or 0, v['slice'].stop, v['slice'].step or 1))
            for k, v in self.area_config.items()
        }

    def _inter_window_attention(self, x):
        b, c, t, f = x.shape
        window_size = min(t, 5)
        num_windows = t // window_size
        original_t = t
        if t % window_size != 0:
            pad_len = window_size - (t % window_size)
            x = F.pad(x, (0, 0, 0, pad_len))
            t = t + pad_len
            num_windows = t // window_size
        x = x.view(b, c, num_windows, window_size, f).permute(0, 3, 1, 2, 4).reshape(b * window_size * c, num_windows, f)
        x = self.inter_window_attn(x, x, x, need_weights=False)[0]
        x = x.reshape(b, window_size, c, num_windows, f).permute(0, 2, 3, 1, 4).reshape(b, c, t, f)
        return x[:, :, :original_t, :]

    def _inter_region_attention(self, x):
        b, c, t, f = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(b * t, c, f)
        global_features = torch.zeros_like(x_flat)
        for _, region_indices in self.region_indices.items():
            region_x = x[:, region_indices, :, :]
            region_global = region_x.mean(dim=1, keepdim=True).permute(0, 2, 1, 3).reshape(b * t, 1, f)
            for idx in region_indices:
                global_features[:, idx:idx + 1, :] = region_global
        x_enhanced = x_flat + self.global_fc(global_features)
        x_attn = self.inter_region_attn(x_enhanced, x_enhanced, x_enhanced, need_weights=False)[0]
        return x_attn.reshape(b, t, c, f).permute(0, 2, 1, 3)

    def _ff_block(self, x):
        b, c, t, f = x.shape
        xr = x.permute(0, 2, 1, 3).reshape(b * t, c, f)
        xr = self.linear2(self.dropout(F.gelu(self.linear1(xr))))
        return xr.reshape(b, t, c, f).permute(0, 2, 1, 3)

    def forward(self, src):
        x = src + self._inter_window_attention(self.norm1(src))
        x = x + self._inter_region_attention(self.norm2(x))
        x = x + self._ff_block(self.norm3(x))
        return x


class CSBrainBackbone(nn.Module):
    def __init__(self, in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8, num_channels=19, brain_regions=None, sorted_indices=None):
        super().__init__()
        del seq_len
        self.num_channels = num_channels
        if sorted_indices is None:
            sorted_indices = list(range(num_channels))
        if brain_regions is None:
            # 5 coarse regions for 19/21ch montages.
            splits = torch.linspace(0, 4, steps=num_channels).round().int().tolist()
            brain_regions = splits

        self.sorted_indices = sorted_indices
        self.area_config = generate_area_config(sorted(brain_regions))

        self.patch_embedding = PatchEmbedding(in_dim=in_dim, d_model=d_model)
        self.tem_embed = TemEmbedEEGLayer(d_model=d_model)
        self.brain_embed = BrainEmbedEEGLayer(d_model=d_model, total_regions=max(brain_regions) + 1)
        self.encoder = nn.ModuleList([
            CSBrainEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, area_config=self.area_config)
            for _ in range(n_layer)
        ])
        self.proj_out = nn.Linear(d_model, out_dim)

    def forward(self, x, mask=None, channel_mask=None, time_mask=None):
        del channel_mask, time_mask
        if x.shape[1] != len(self.sorted_indices):
            idx = self.sorted_indices[: x.shape[1]]
            x = x[:, idx, :, :]
        else:
            x = x[:, self.sorted_indices, :, :]

        patch_emb = self.patch_embedding(x, mask)
        for layer in self.encoder:
            patch_emb = self.tem_embed(patch_emb) + patch_emb
            patch_emb = self.brain_embed(patch_emb, self.area_config) + patch_emb
            patch_emb = layer(patch_emb)
        return self.proj_out(patch_emb)
