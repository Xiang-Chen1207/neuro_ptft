import torch
import torch.nn as nn
from models.wrapper import CBraModWrapper

class CBraModWrapperDynamic(CBraModWrapper):
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, C, N, P) input tensor
            mask: (B, N) boolean tensor where True indicates padding (to be ignored)
        """
        if self.task_type == 'classification':
            # Backbone forward
            # We don't pass mask to backbone because it expects pretraining mask (random drop)
            # and doesn't handle padding mask in the encoder currently.
            # Transformer will process padding tokens, but we filter them out at pooling.
            feats = self.backbone(x) # (B, C, N, D)
            return self._forward_classification_masked(feats, mask)
        else:
            return super().forward(x, mask)

    def _forward_classification_masked(self, feats, mask):
        # feats: (B, C, N, D)
        # mask: (B, N) where True indicates padding
        
        # Check for custom heads (Flagship / FeatOnly)
        if getattr(self, 'feat_query', None) is not None:
            # 1. Compute Cross Attn Features (Feat Path)
            # Flatten channel and time for attention: (B, C*N, D)
            B, C, N, D = feats.shape
            feats_flat = feats.view(B, C * N, D)
            
            # Apply Mask to Cross Attention if needed
            # mask is (B, N). We need to expand it to (B, C*N)
            # True means padding (ignored)
            if mask is not None:
                # mask: (B, N) -> (B, 1, N) -> (B, C, N) -> (B, C*N)
                attn_mask = mask.unsqueeze(1).expand(-1, C, -1).reshape(B, C * N)
                # MultiheadAttention key_padding_mask expects True for ignored positions
            else:
                attn_mask = None
                
            query = self.feat_query.expand(B, -1, -1)
            # Pass key_padding_mask to ignore padded tokens
            attn_output, _ = self.feat_attn(query, feats_flat, feats_flat, key_padding_mask=attn_mask)
            feat_token = attn_output.squeeze(1) # (B, 200)

            if hasattr(self, 'fc_norm') and isinstance(self.fc_norm, nn.LayerNorm) and hasattr(self, 'feat_head_mlp') and self.head.in_features == 400:
                 # Flagship Concat
                 # Compute Masked Global Average Pooling for EEG part
                 if mask is not None:
                    feats_c = feats.mean(dim=1) 
                    valid_mask = (~mask).float().unsqueeze(-1) # (B, N, 1)
                    feats_masked = feats_c * valid_mask 
                    sum_feats = feats_masked.sum(dim=1) # (B, D)
                    valid_count = valid_mask.sum(dim=1) # (B, 1)
                    valid_count = torch.clamp(valid_count, min=1.0)
                    pooled = sum_feats / valid_count
                 else:
                    pooled = feats.mean(dim=[1, 2])
                    
                 eeg_feat = self.fc_norm(pooled) # (B, 200)
                 concat_feat = torch.cat([eeg_feat, feat_token], dim=1) # (B, 400)
                 return self.head(concat_feat)
                 
            elif hasattr(self, 'feat_head_mlp'):
                 # FeatOnly CrossAttn
                 feat_out = self.feat_head_mlp(feat_token) # (B, feature_dim)
                 return self.head(feat_out)
        
        if isinstance(self.fc_norm, nn.LayerNorm):
            if mask is not None:
                # 1. Mean over channels (B, N, D)
                # We assume all channels are valid for a valid time step
                feats_c = feats.mean(dim=1) 
                
                # 2. Mean over time (N) with mask
                # Invert mask: 1 for valid, 0 for padding
                valid_mask = (~mask).float().unsqueeze(-1) # (B, N, 1)
                
                # Zero out padding tokens
                feats_masked = feats_c * valid_mask 
                
                # Sum over time
                sum_feats = feats_masked.sum(dim=1) # (B, D)
                
                # Count valid tokens
                valid_count = valid_mask.sum(dim=1) # (B, 1)
                valid_count = torch.clamp(valid_count, min=1.0)
                
                pooled = sum_feats / valid_count
            else:
                pooled = feats.mean(dim=[1, 2])
                
            out = self.head(self.fc_norm(pooled))
        else:
            # For non-pooling heads, we just pass feats. 
            # Note: Flatten heads might fail with variable length if they expect fixed size.
            out = self.head(feats)
            
        if self.num_classes == 1:
            out = out.view(-1)
        return out
