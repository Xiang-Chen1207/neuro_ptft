import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.config_mamba import MambaConfig
from .modules.mixer_seq_simple import MixerModel
from torch.nn.modules import TransformerEncoderLayer, TransformerEncoder
from einops import rearrange
from .eegmamba import EEGMambaBackbone, PatchEmbedding, _weights_init

class EEGMambaBackbonePrefix(EEGMambaBackbone):
    def __init__(self, in_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8, feature_dim=0, num_feature_tokens=1):
        super().__init__(in_dim, d_model, dim_feedforward, seq_len, n_layer, nhead)
        
        # Override config to use Mamba1 for stability with Triton 3.x
        config = MambaConfig()
        config.d_model = d_model
        config.n_layer = n_layer
        config.d_intermediate = 0
        config.rms_norm = False
        config.fused_add_norm = False
        config.ssm_cfg = {
            "layer": "Mamba1",  # Changed from Mamba2 to Mamba1
            # "headdim": 50,    # Mamba1 does not support headdim
            "d_state": 64,
        }
        # Re-initialize encoder with Mamba1
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

        # Feature Tokens (learnable parameters)
        self.feature_dim = feature_dim
        self.num_feature_tokens = num_feature_tokens
        
        if feature_dim > 0:
            self.feat_tokens = nn.Parameter(
                torch.zeros(1, num_feature_tokens, d_model)
            )
            nn.init.normal_(self.feat_tokens, std=0.02)
    
    def forward(self, x, mask=None, channel_mask=None, time_mask=None):
        bz, ch_num, seq_len, patch_size = x.shape
        if channel_mask is not None:
            x = x.masked_fill(channel_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
        if time_mask is not None:
            x = x.masked_fill(time_mask.unsqueeze(1).unsqueeze(-1), 0.0)
            
        hidden_states = self.patch_embedding(x, mask=mask)
        hidden_states = rearrange(hidden_states, 'b c l d -> b (c l) d')
        
        # --- Prepend Feature Tokens ---
        feat_out = None
        if self.feature_dim > 0:
            feat_tokens = self.feat_tokens.expand(bz, -1, -1)
            hidden_states = torch.cat([feat_tokens, hidden_states], dim=1)
        
        # Mamba processing (feature tokens and EEG tokens together)
        hidden_states = self.encoder(hidden_states)
        
        # --- Split Feature Tokens ---
        if self.feature_dim > 0:
            feat_out = hidden_states[:, :self.num_feature_tokens, :]
            hidden_states = hidden_states[:, self.num_feature_tokens:, :]
        
        hidden_states = rearrange(hidden_states, 'b (c l) d -> b c l d', l=seq_len)
        return hidden_states, feat_out
