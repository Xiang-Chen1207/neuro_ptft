import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from .backbone import CBraModBackbone
from .csbrain_backbone import CSBrainBackbone
try:
    from .eegmamba import EEGMambaBackbone
except ImportError:
    EEGMambaBackbone = None
try:
    from .eegmamba_prefix import EEGMambaBackbonePrefix
except ImportError:
    EEGMambaBackbonePrefix = None
try:
    from .st_eegformer_mae import ST_EEGFormer_MAE
except ImportError:
    ST_EEGFormer_MAE = None
try:
    from .reve import ReveBackbone
except ImportError:
    ReveBackbone = None
try:
    from .tech.backbone import TeChBackbone
except ImportError:
    TeChBackbone = None

class CBraModWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_config = config.get('model', {})
        self._init_backbone(model_config)
        self._init_task_heads(config, model_config)
        self._load_pretrained_weights(model_config)

    def _init_backbone(self, model_config):
        model_name = model_config.get('name', 'cbramod')
        
        if model_name == 'eegmamba':
             self.backbone = EEGMambaBackbone(
                in_dim=model_config.get('in_dim', 200),
                d_model=model_config.get('d_model', 200),
                dim_feedforward=model_config.get('dim_feedforward', 800),
                seq_len=model_config.get('seq_len', 30),
                n_layer=model_config.get('n_layer', 12),
                nhead=model_config.get('nhead', 8)
            )
        elif model_name == 'eegmamba_prefix':
             if EEGMambaBackbonePrefix is None:
                 raise ImportError("EEGMambaBackbonePrefix could not be imported. Check for circular imports or missing file.")
             self.backbone = EEGMambaBackbonePrefix(
                in_dim=model_config.get('in_dim', 200),
                d_model=model_config.get('d_model', 200),
                dim_feedforward=model_config.get('dim_feedforward', 800),
                seq_len=model_config.get('seq_len', 30),
                n_layer=model_config.get('n_layer', 12),
                nhead=model_config.get('nhead', 8),
                feature_dim=model_config.get('feature_dim', 0),
                num_feature_tokens=model_config.get('num_feature_tokens', 1)
            )
        elif model_name == 'st_eegformer_mae':
             self.backbone = ST_EEGFormer_MAE(
                in_dim=model_config.get('in_dim', 200),
                d_model=model_config.get('d_model', 512),
                depth=model_config.get('n_layer', 8),
                nhead=model_config.get('nhead', 8),
                mlp_ratio=model_config.get('mlp_ratio', 4.0),
                decoder_depth=model_config.get('decoder_depth', 4),
                decoder_embed_dim=model_config.get('decoder_embed_dim', 384),
                decoder_num_heads=model_config.get('decoder_num_heads', 16),
                seq_len=model_config.get('seq_len', 60),
                num_channels=model_config.get('num_channels', 20)
             )
        elif model_name == 'reve':
             self.backbone = ReveBackbone(
                in_dim=model_config.get('in_dim', 200),
                d_model=model_config.get('d_model', 512),
                seq_len=model_config.get('seq_len', 60),
                n_layer=model_config.get('n_layer', 12),
                nhead=model_config.get('nhead', 8),
                mlp_dim_ratio=model_config.get('mlp_dim_ratio', 4.0),
                freqs=model_config.get('freqs', 4),
                noise_ratio=model_config.get('noise_ratio', 0.0025),
                patch_overlap=model_config.get('patch_overlap', 0)
             )
        elif model_name == 'tech':
             if TeChBackbone is None:
                 raise ImportError("TeChBackbone could not be imported.")
             self.backbone = TeChBackbone(**model_config)
        elif model_name == 'csbrain':
             self.backbone = CSBrainBackbone(
                in_dim=model_config.get('in_dim', 200),
                out_dim=model_config.get('out_dim', 200),
                d_model=model_config.get('d_model', 200),
                dim_feedforward=model_config.get('dim_feedforward', 800),
                seq_len=model_config.get('seq_len', 30),
                n_layer=model_config.get('n_layer', 12),
                nhead=model_config.get('nhead', 8),
                num_channels=model_config.get('num_channels', 19),
                brain_regions=model_config.get('brain_regions'),
                sorted_indices=model_config.get('sorted_indices'),
            )
        else:
            self.backbone = CBraModBackbone(
                in_dim=model_config.get('in_dim', 200),
                d_model=model_config.get('d_model', 200),
                dim_feedforward=model_config.get('dim_feedforward', 800),
                seq_len=model_config.get('seq_len', 30),
                n_layer=model_config.get('n_layer', 12),
                nhead=model_config.get('nhead', 8)
            )

    def _init_task_heads(self, config, model_config):
        self.task_type = config.get('task_type', 'classification')
        self.num_classes = model_config.get('num_classes', 2)
        self.dropout = model_config.get('dropout', 0.1)
        self.d_model = model_config.get('d_model', 200)
        
        # Pretraining Config
        self.pretrain_tasks = model_config.get('pretrain_tasks', ['reconstruction'])
        self.feature_token_type = model_config.get('feature_token_type', 'gap')
        self.feature_token_strategy = model_config.get('feature_token_strategy', 'single')
        self.feature_group_count = model_config.get('feature_group_count', 5)
        
        if self.task_type == 'pretraining':
            self._init_pretraining_heads(model_config)
        else:
            self._init_classification_head(model_config)

    def _init_pretraining_heads(self, model_config):
        # Reconstruction Head
        if 'reconstruction' in self.pretrain_tasks:
            self.head = nn.Linear(self.d_model, model_config.get('out_dim', 200))
        else:
            self.head = None
        
        # Feature Prediction Head
        self.feature_dim = model_config.get('feature_dim', 0)
        if 'feature_pred' in self.pretrain_tasks and self.feature_dim > 0:
            if self.feature_token_type == 'cross_attn':
                self._init_cross_attn_feature_head()
            elif self.feature_token_type == 'prefix':
                self._init_prefix_feature_head()
            else:
                self._init_gap_feature_head()
        else:
            self.feature_head = None

    def _init_cross_attn_feature_head(self):
        # Determine tokens based on strategy
        if self.feature_token_strategy == 'single':
            num_tokens, out_dim = 1, self.feature_dim
        elif self.feature_token_strategy == 'all':
            num_tokens, out_dim = self.feature_dim, 1
        elif self.feature_token_strategy == 'group':
            num_tokens, out_dim = self.feature_group_count, None # Handled by MLP
        else:
            raise ValueError(f"Unknown strategy: {self.feature_token_strategy}")
        
        self.feat_query = nn.Parameter(torch.zeros(1, num_tokens, self.d_model))
        nn.init.normal_(self.feat_query, std=0.02)
        self.feat_attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=4, batch_first=True)
        
        if self.feature_token_strategy == 'group':
            self.feature_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_tokens * self.d_model, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.feature_dim)
            )
        else:
            self.feature_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, out_dim)
            )

    def _init_gap_feature_head(self):
        self.feature_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.feature_dim)
        )

    def _init_prefix_feature_head(self):
        self.feature_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.feature_dim)
        )

    def _init_classification_head(self, model_config):
        head_type = model_config.get('head_type', 'flatten')
        
        if head_type == 'feat_cross_attn':
            # For FeatOnly: Use Cross Attention like in Pretraining
            self.fc_norm = nn.Identity()
            self.feature_dim = model_config.get('feature_dim', 768)
            self.feature_token_strategy = 'single'
            
            self.feat_query = nn.Parameter(torch.zeros(1, 1, self.d_model))
            nn.init.normal_(self.feat_query, std=0.02)
            
            self.feat_attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=4, batch_first=True)
            
            self.feat_head_mlp = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.feature_dim)
            )
            
            self.head = nn.Linear(self.feature_dim, self.num_classes)
            
        elif head_type == 'flagship_concat':
            # For Flagship: 200dim eeg + 200dim feat -> 400 -> Classes
            self.fc_norm = nn.LayerNorm(self.d_model)
            
            self.feature_dim = model_config.get('feature_dim', 768)
            self.feature_token_strategy = 'single'
            self.feat_query = nn.Parameter(torch.zeros(1, 1, self.d_model))
            nn.init.normal_(self.feat_query, std=0.02)
            self.feat_attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=4, batch_first=True)
            self.feat_head_mlp = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.feature_dim)
            )
            
            self.head = nn.Linear(self.d_model * 2, self.num_classes)
            
        elif head_type == 'pooling':
            self.fc_norm = nn.LayerNorm(self.d_model)
            self.head = nn.Linear(self.d_model, self.num_classes)
        else:
            self.fc_norm = nn.Identity()
            self.head = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.LazyLinear(10 * 200),
                nn.ELU(),
                nn.Dropout(self.dropout),
                nn.Linear(10 * 200, 200),
                nn.ELU(),
                nn.Dropout(self.dropout),
                nn.Linear(200, self.num_classes),
            )

    def _load_pretrained_weights(self, model_config):
        # We handle weight loading explicitly in the training script
        # So we skip automatic loading here if not desired, 
        # BUT the config usually has 'use_pretrained': true for the baseline backbone.
        
        # If the user script calls load_pretrained_model(), it will overwrite these anyway.
        # But to be safe and avoid double loading or confusion, we can keep it.
        # However, for FeatOnly/Flagship, we load specific checkpoints later.
        
        if not model_config.get('use_pretrained', False):
            return
            
        pretrained_path = model_config.get('pretrained_path')
        if not pretrained_path:
            return
            
        self.load_pretrained(pretrained_path)

    def load_pretrained(self, checkpoint_path):
        print(f"Loading pretrained weights from {checkpoint_path}")
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except TypeError:
             # Fallback for older torch versions that don't support weights_only arg
             state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle 'model' key wrapper if present
        if 'model' in state_dict:
            state_dict = state_dict['model']
            
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('proj_out'): # Skip projection head from pretraining
                continue
                
            # If checkpoint has 'patch_embedding' etc directly, map to 'backbone.patch_embedding'
            if not k.startswith('backbone.'):
                # Check if it belongs to backbone components
                if any(k.startswith(p) for p in ['patch_embedding', 'encoder']):
                    new_key = f"backbone.{k}"
                else:
                    new_key = k # Might be other keys or unexpected
            else:
                new_key = k
                
            new_state_dict[new_key] = v
            
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        print(f"Missing keys: {missing}")
        # print(f"Unexpected keys: {unexpected}")

    def forward(self, x, mask=None, channel_mask=None, time_mask=None):
        if self.task_type == 'pretraining':
            return self._forward_pretraining(x, mask, channel_mask, time_mask)
        else:
            return self._forward_classification(x, channel_mask, time_mask)

    def _forward_pretraining(self, x, mask=None, channel_mask=None, time_mask=None):
        if ST_EEGFormer_MAE is not None and isinstance(self.backbone, ST_EEGFormer_MAE):
            # MAE handles masking and reconstruction internally
            pred, mask = self.backbone(x, channel_mask=channel_mask, time_mask=time_mask)
            return pred, mask, None

        if mask is None:
            mask = self._generate_mask(x, channel_mask=channel_mask, time_mask=time_mask)
        
        # Check if backbone is prefix version
        if hasattr(self.backbone, 'feature_dim') and isinstance(self.backbone, EEGMambaBackbonePrefix):
             feats, feat_tokens_out = self.backbone(x, mask, channel_mask=channel_mask, time_mask=time_mask)
        else:
             feats = self.backbone(x, mask, channel_mask=channel_mask, time_mask=time_mask)
             feat_tokens_out = None
        
        # Reconstruction Output
        out = self.head(feats) if self.head is not None else None
        
        # Feature Prediction Output
        feature_pred = None
        if getattr(self, 'feature_head', None) is not None:
            if self.feature_token_type == 'cross_attn':
                feature_pred = self._forward_cross_attn_head(feats)
            elif self.feature_token_type == 'prefix' and feat_tokens_out is not None:
                if feat_tokens_out.shape[1] == 1:
                     feature_pred = self.feature_head(feat_tokens_out.squeeze(1))
                else:
                     feature_pred = self.feature_head(feat_tokens_out.flatten(1))
            else:
                feature_pred = self._forward_gap_head(feats)
                
        return out, mask, feature_pred

    def _forward_classification(self, x, channel_mask=None, time_mask=None):
        feats = self.backbone(x, channel_mask=channel_mask, time_mask=time_mask)
        
        # Check for custom heads
        if getattr(self, 'feat_query', None) is not None:
            # 1. Compute Cross Attn Features (Feat Path)
            if feats.ndim == 2:
                # (B, D) -> (B, 1, D)
                feats_flat = feats.unsqueeze(1)
                B, D = feats.shape
            else:
                B, C, N, D = feats.shape
                feats_flat = feats.view(B, C * N, D)
            
            query = self.feat_query.expand(B, -1, -1)
            attn_output, _ = self.feat_attn(query, feats_flat, feats_flat)
            feat_token = attn_output.squeeze(1) # (B, 200)

            if hasattr(self, 'fc_norm') and isinstance(self.fc_norm, nn.LayerNorm) and hasattr(self, 'feat_head_mlp') and self.head.in_features == 400:
                 # Flagship Concat
                 if feats.ndim == 4:
                    pooled = feats.mean(dim=[1, 2])
                 elif feats.ndim == 3:
                    pooled = feats.mean(dim=1)
                 else:
                    pooled = feats
                 
                 eeg_feat = self.fc_norm(pooled) # (B, 200)
                 concat_feat = torch.cat([eeg_feat, feat_token], dim=1) # (B, 400)
                 return self.head(concat_feat)
                 
            elif hasattr(self, 'feat_head_mlp'):
                 # FeatOnly CrossAttn
                 feat_out = self.feat_head_mlp(feat_token) # (B, feature_dim)
                 return self.head(feat_out)
        
        if isinstance(self.fc_norm, nn.LayerNorm):
            # Global Average Pooling
            if feats.ndim == 4:
                pooled = feats.mean(dim=[1, 2])
            elif feats.ndim == 3:
                pooled = feats.mean(dim=1)
            else:
                pooled = feats
            out = self.head(self.fc_norm(pooled))
        else:
            out = self.head(feats)
            
        if self.num_classes == 1:
            out = out.view(-1)
        return out

    def _generate_mask(self, x, channel_mask=None, time_mask=None):
        # Generate random mask (50% masking)
        B, C, N, P = x.shape
        mask = torch.bernoulli(torch.full((B, C, N), 0.5, device=x.device)).to(x.device)
        
        # Ensure padded regions are NOT masked (mask=0) so we don't compute loss on them
        # channel_mask: (B, C) - True if padded
        # time_mask: (B, N) - True if padded
        if channel_mask is not None:
            # (B, C, 1)
            mask = mask.masked_fill(channel_mask.unsqueeze(-1), 0.0)
        if time_mask is not None:
            # (B, 1, N)
            mask = mask.masked_fill(time_mask.unsqueeze(1), 0.0)
            
        return mask

    def _forward_cross_attn_head(self, feats):
        # feats: (B, C, N, D) -> (B, S, D) where S = C*N
        if feats.ndim == 2:
            B, D = feats.shape
            feats_flat = feats.unsqueeze(1)
        else:
            B, C, N, D = feats.shape
            feats_flat = feats.reshape(B, C * N, D)
        
        # Expand query: (B, num_tokens, D)
        query = self.feat_query.expand(B, -1, -1)
        
        # Cross Attention
        attn_output, _ = self.feat_attn(query, feats_flat, feats_flat)
        
        if self.feature_token_strategy == 'single':
            return self.feature_head(attn_output).squeeze(1)
        elif self.feature_token_strategy == 'all':
            return self.feature_head(attn_output).squeeze(-1)
        elif self.feature_token_strategy == 'group':
            return self.feature_head(attn_output)
        return None

    def _forward_gap_head(self, feats):
        if feats.ndim == 4:
            global_feat = feats.mean(dim=[1, 2])
        elif feats.ndim == 3:
            global_feat = feats.mean(dim=1)
        else:
            global_feat = feats
        return self.feature_head(global_feat)
