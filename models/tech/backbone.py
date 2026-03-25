import torch
import torch.nn as nn
from types import SimpleNamespace
from .model import TeChModel

class TeChBackbone(nn.Module):
    def __init__(self, in_dim=200, d_model=200, seq_len=30, n_layer=2, 
                 feature_dim=0, num_feature_tokens=1, **kwargs):
        super().__init__()
        
        # Adapt kwargs to TeCh Config
        # TeCh Config expects:
        # v_layer, t_layer, d_model, num_class, augmentations, seq_len, patch_len, enc_in, n_heads, dropout
        
        # Mapping ptft config to TeCh config
        # in_dim in ptft usually is patch_size (200). 
        # But TeCh expects raw sequence length as seq_len?
        # In ptft:
        # Input x is (B, C, N, P). Total time = N * P.
        # TeCh expects (B, T, C). T = N * P.
        # So enc_in should be C (channels).
        # seq_len in TeCh config should be Total Time T.
        
        # However, ptft 'seq_len' usually refers to N (number of patches).
        # And 'in_dim' refers to P (patch size).
        
        # Let's infer params
        # We need 'num_channels' which is not always passed to backbone __init__ in ptft unless specified.
        # CBraModWrapper passes: in_dim, d_model, dim_feedforward, seq_len, n_layer, nhead.
        # It does NOT pass num_channels by default unless we add it to config.
        # ST_EEGFormer_MAE receives num_channels.
        
        self.patch_size = in_dim # 200
        self.num_patches = seq_len # 30
        self.total_time = self.patch_size * self.num_patches # 6000
        
        # We assume a default channel count if not provided, but TeCh needs it for Conv2d kernel.
        # We MUST ensure enc_in is correct.
        # In ptft, data varies (19, 21, 62 channels).
        # If enc_in is fixed in model, it won't work for different datasets without rebuilding.
        # But TeCh CrossChannelPatching uses kernel_size=(enc_in, patch_len).
        # This implies TeCh is specific to channel count.
        
        # WORKAROUND:
        # If TeCh requires fixed channel count, we must set it.
        # Default to 20ish? Or allow dynamic?
        # Conv2d kernel size can be dynamic if we recreate it, but that's messy.
        # If we use `num_channels` from kwargs, we can set it.
        self.enc_in = kwargs.get('num_channels', 20) # Default to 20 if not set
        
        config_dict = {
            'v_layer': n_layer, # Use n_layer for both? Or split?
            't_layer': n_layer,
            'd_model': d_model,
            'num_class': 0, # Not used in features mode
            'augmentations': kwargs.get('augmentations', 'none'),
            'seq_len': self.total_time,
            'patch_len': kwargs.get('patch_len', 200), # TeCh patch len, maybe different from ptft patch size
            'enc_in': self.enc_in,
            'n_heads': kwargs.get('nhead', 8),
            'dropout': kwargs.get('dropout', 0.1)
        }
        
        self.configs = SimpleNamespace(**config_dict)
        self.model = TeChModel(self.configs)
        
    def forward(self, x, mask=None, channel_mask=None, time_mask=None):
        # x: (B, C, N, P)
        B, C, N, P = x.shape
        
        # TeCh expects (B, T, C)
        # Reshape: (B, C, N, P) -> (B, C, N*P) -> (B, N*P, C)
        x_flat = x.view(B, C, N * P).transpose(1, 2) # (B, T, C)
        
        # Handle Channel Mismatch
        # If input C != self.enc_in, TeCh will crash in CrossChannelPatching (Conv2d kernel height).
        # We can pad or slice if necessary, or throw error.
        if C != self.enc_in:
            # print(f"Warning: Input channels {C} != Model channels {self.enc_in}")
            if C < self.enc_in:
                # Pad with zeros
                padding = torch.zeros(B, N*P, self.enc_in - C, device=x.device, dtype=x.dtype)
                x_flat = torch.cat([x_flat, padding], dim=2)
            else:
                # Slice
                x_flat = x_flat[:, :, :self.enc_in]
        
        # Detect if we are in reconstruction mode via kwargs or config?
        # Backbone forward signature doesn't pass task_type easily.
        # But we can infer if we need sequence output if mask is provided?
        # In ptft, mask is provided for pretraining.
        
        # However, CBraModWrapper always calls backbone(x, mask, ...)
        # We can default to return features, unless we hack it.
        # But wait, CBraModWrapper._forward_pretraining expects 'feats'.
        # If 'reconstruction' is in tasks, 'head' will be applied to 'feats'.
        # If 'feats' is (B, D), head output is (B, out_dim).
        # If 'feats' is (B, N, D), head output is (B, N, out_dim).
        
        # Let's try to return sequence if mask is not None (Pretraining usually has mask)
        # But Feature Prediction task also has mask.
        # If we return sequence, Feature Prediction head (GAP) will handle it (mean dim 1).
        # So it is safe to return sequence for pretraining!
        
        return_seq = (mask is not None)
        
        if return_seq:
             seq_feats = self.model(x_flat, return_seq=True) # (B, N, D)
             
             if seq_feats is None:
                 # Fallback
                 return self.model(x_flat)
             
             # Now we have (B, N, D).
             # Reconstruction target x is (B, C, N, P).
             # If we return (B, N, D), wrapper head produces (B, N, P).
             # This mismatches (B, C, N, P).
             # We must expand to (B, C, N, D).
             
             # Expand: (B, N, D) -> (B, 1, N, D) -> (B, C, N, D)
             # Note: Using original C from input, not self.enc_in (which might be padded/sliced)
             seq_feats_expanded = seq_feats.unsqueeze(1).expand(-1, C, -1, -1)
             
             return seq_feats_expanded
        else:
             feats = self.model(x_flat) # (B, D)
             return feats
