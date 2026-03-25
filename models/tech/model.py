import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from .layers.transformer_encdec import Encoder, EncoderLayer
from .layers.augmentation import get_augmentation
import math

class TeChModel(nn.Module):
    def __init__(self, configs):
        super(TeChModel, self).__init__()
        self.v_layer = configs.v_layer
        self.t_layer = configs.t_layer
        self.channel_encoder = nn.Sequential(
            Aug_Channel_Embedding(configs),
            Encoder([EncoderLayer(configs) for _ in range(configs.v_layer)])
        ) if configs.v_layer > 0 else nn.Identity()
        
        self.temporal_encoder = nn.Sequential(
            Aug_Temporal_Embedding(configs),
            Encoder([EncoderLayer(configs) for _ in range(configs.t_layer)])
        ) if configs.t_layer > 0 else nn.Identity()
        
        # Projector removed or optional, as we want features
        # self.projector = nn.Linear(configs.d_model, configs.num_class)

    def forward(self, x_enc, return_seq=False):
        # x_enc: (B, T, C) or (B, SeqLen, EncIn)
        B, T, N = x_enc.shape
        
        channel_out = 0
        if self.v_layer > 0:
            # channel_encoder returns (B, C, D) ?
            # Aug_Channel_Embedding returns (B, C, D)
            # Encoder maintains shape
            channel_enc_out = self.channel_encoder(x_enc)
            channel_out = channel_enc_out.mean(1) # Global Mean Pooling over Channels -> (B, D)
            
        temporal_out = 0
        temporal_seq = None
        if self.t_layer > 0:
            # Aug_Temporal_Embedding returns (B, T_patches, D)
            # Encoder maintains shape
            temporal_seq = self.temporal_encoder(x_enc)
            temporal_out = temporal_seq.mean(1) # Global Mean Pooling over Time -> (B, D)
        
        # Combine
        features = channel_out + temporal_out # (B, D)
        
        if return_seq:
             # Return Temporal Sequence for Reconstruction
             # If temporal_encoder is not active, return None or zeros
             if temporal_seq is not None:
                 # Ensure temporal_seq shape is (B, N, D)
                 # temporal_encoder output:
                 # CrossChannelPatching returns (B, T_patches, D)
                 # Encoder maintains shape
                 
                 # Verify shape against N
                 # x_enc was (B, T_total, C)
                 # T_total = N * P
                 # T_patches = T_total // P = N
                 # So temporal_seq should be (B, N, D)
                 
                 return temporal_seq
             else:
                 # Fallback if only channel encoder is active?
                 # But channel encoder output is (B, C, D), temporal is (B, T, D).
                 # Reconstruction usually needs T.
                 return None

        return features


class Aug_Channel_Embedding(nn.Module):
    def __init__(self, configs):
        super().__init__()
        aug_idxs = configs.augmentations.split(",")
        self.augmentation = nn.ModuleList(
            [get_augmentation(aug) for aug in aug_idxs]
        )
        # Replaced fixed Linear with Patch-like or Time-Independent Projection
        # To support variable length T, we should not project T -> d_model via Linear(T, D).
        # We want to embed channels.
        # Option A: Linear(T, D) -> Requires fixed T. (Original TeCh)
        # Option B: Conv1d(T, D, kernel=1) -> Pointwise? No, we want to aggregate time?
        # Option C: Use PatchEmbedding logic like EEGMamba?
        
        # TeCh's "Channel Embedding" is essentially treating each Channel as a token, and the Time dimension as the feature vector.
        # It projects the Time vector (T) to a hidden dimension (d_model).
        # x: (B, C, T) -> Linear(T, D) -> (B, C, D).
        # This creates C tokens, each with dim D.
        # This IS dependent on T.
        
        # To support variable T, we can:
        # 1. Resize input T to a fixed size (Interpolation).
        # 2. Use a T-independent projection (e.g. GAP + Linear).
        # 3. Use Patching on T?
        
        # If we want to follow TeCh philosophy: "Channel Embedding" captures the global temporal pattern of a channel.
        # The easiest way to adapt to variable T is to Interpolate T -> fixed_T before Linear.
        # This preserves the "shape" of the waveform while normalizing length.
        
        self.seq_len = configs.seq_len # Fixed target length (e.g. 12000 or 6000)
        self.Channel_Embedding = nn.Linear(self.seq_len, configs.d_model)
        
        # Positional Embedding for Channels?
        # Original TeCh uses PositionalEmbedding(d_model=configs.seq_len).
        # And adds it to x (B, C, T).
        # This means it adds position info to TIME steps before projection.
        self.pos_emb = PositionalEmbedding(d_model=self.seq_len)

    def forward(self, x):  # (batch_size, seq_len, enc_in) -> (B, T, C) input
        x = x.transpose(1, 2)  # (batch_size, enc_in, seq_len) -> (B, C, T)
        if self.training and len(self.augmentation) > 0:
            aug_idx = random.randint(0, len(self.augmentation) - 1)
            x_aug = self.augmentation[aug_idx](x)
        else:
            x_aug = x
            
        # DYNAMIC ADAPTATION:
        # Interpolate x_aug time dimension to match self.seq_len
        if x_aug.shape[2] != self.seq_len:
             x_aug = F.interpolate(x_aug, size=self.seq_len, mode='linear', align_corners=False)
             
        # Add Positional Embedding (Time)
        # pos_emb(x_aug) should return (1, 1, T) or (1, C, T)?
        # Our PositionalEmbedding returns pe[:, :, :T].
        # pe shape is (1, d_model, max_len) or similar?
        # Let's check PositionalEmbedding implementation below.
        
        x_aug = x_aug + self.pos_emb(x_aug)
        
        # Projection: (B, C, T_fixed) -> (B, C, D)
        return self.Channel_Embedding(x_aug)

class Aug_Temporal_Embedding(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.patch_len = configs.patch_len
        aug_idxs = configs.augmentations.split(",")
        self.augmentation = nn.ModuleList(
            [get_augmentation(aug) for aug in aug_idxs]
        )
        self.Temporal_Embedding = CrossChannelPatching(configs) if self.patch_len > 1 else nn.Linear(configs.enc_in, configs.d_model)
        
        # Temporal Embedding in TeCh projects (B, C, T) -> (B, T_patches, D).
        # It treats Time Patches as tokens.
        # This path NATURALLY supports variable T, because Conv2d slides over T.
        # Output number of patches N depends on T.
        
        # However, it ADDS Positional Embedding to x BEFORE patching?
        # "self.pos_emb = PositionalEmbedding(d_model=configs.seq_len)"
        # "x_aug = x_aug + self.pos_emb(x_aug)"
        # If pos_emb is fixed length, it crashes for variable T.
        
        # We need a PositionalEmbedding that supports variable T.
        # Our modified PositionalEmbedding slices to input length.
        # But we need to ensure max_len is large enough.
        
        self.pos_emb = PositionalEmbedding(d_model=configs.seq_len, max_len=max(50000, configs.seq_len)) 
        # Increased max_len to handle long sequences safely.
        # Note: d_model here is actually channel dim? Or Time dim?
        # In TeCh original: PositionalEmbedding(d_model=configs.seq_len).
        # pe shape: (1, seq_len, max_len)? No.
        # PositionalEmbedding(d_model, max_len).
        # pe = zeros(max_len, d_model).
        # It seems TeCh's PositionalEmbedding implementation is confusing d_model and len.
        
        # Let's look at PositionalEmbedding again.
        # pe = zeros(max_len, d_model).
        # forward returns pe[:, :x.size(1)]. -> Slices first dim of pe (max_len).
        # So max_len must cover the sequence length dimension of input x.
        # If input is (B, C, T), and we want to add to T?
        # Original forward: pe[:, :x.size(1)]. x.size(1) is C.
        # So it slices C?
        # But TeCh initialized d_model=configs.seq_len (T).
        # So pe is (max_len, T).
        # If it slices C, it returns (C, T).
        # Then (1, C, T) is added to (B, C, T).
        # This adds a "Channel Positional Encoding" to the T-vector?
        # "Channel 0 gets pe[0]", "Channel 1 gets pe[1]".
        # Yes, this seems to be Channel Embedding (encoding channel index).
        
        # So for Aug_Temporal_Embedding, do we want Channel Embedding?
        # TeCh code uses the same PositionalEmbedding logic.
        # So it adds Channel Position info before merging channels via CrossChannelPatching.
        
        # Wait, if it adds Channel Position, then it doesn't depend on T length?
        # pe is (max_len, T).
        # We slice max_len to C.
        # Result is (C, T).
        # This requires T to be fixed (d_model size).
        
        # FIX for Variable T in Temporal Path:
        # We cannot use fixed T in pe.
        # We need pe to be (max_len, max_T).
        # Or generate sinusoidal on the fly.
        
        # Since we are adding Channel Position, maybe we should just use learnable channel embedding?
        # Or interpolate the T dimension of the PE.
        
        # Strategy:
        # 1. Initialize PE with a large enough max_len (for channels) and max_d_model (for Time).
        # But d_model is fixed at init.
        
        # If we use interpolation for PE:
        # pe is (1, C_max, T_fixed).
        # Input is (B, C, T_current).
        # We slice C. We interpolate T_fixed -> T_current.
        
        pass 

    def forward(self, x):  # (batch_size, seq_len, enc_in)
        x = x.transpose(1, 2)  # (batch_size, enc_in, seq_len)
        if self.training and len(self.augmentation) > 0:
            aug_idx = random.randint(0, len(self.augmentation) - 1)
            x_aug = self.augmentation[aug_idx](x)
        else:
            x_aug = x
            
        # Handle PE for Variable T
        # x_aug: (B, C, T)
        # pos_emb.pe: (1, C_max, T_fixed)
        # We need (1, C, T).
        
        pe = self.pos_emb.pe # (1, max_len, d_model=T_fixed)
        
        # Slice Channels (dim 1)
        if x_aug.size(1) > pe.size(1):
             # Pad PE if more channels than expected? Or just slice input?
             # Assuming max_len is large enough (5000 default).
             pass
        pe_sliced = pe[:, :x_aug.size(1), :] # (1, C, T_fixed)
        
        # Interpolate Time (dim 2)
        if pe_sliced.size(2) != x_aug.size(2):
             pe_sliced = F.interpolate(pe_sliced, size=x_aug.size(2), mode='linear', align_corners=False)
             
        x_aug = x_aug + pe_sliced # Adds (1, C, T) to (B, C, T)
        
        if self.patch_len == 1: 
            x_aug = x_aug.transpose(1, 2) # (B, T, C)
            # Linear maps C -> d_model. Result (B, T, d_model).
            
        return self.Temporal_Embedding(x_aug)
    
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False 

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()
        
        # Handle d_model odd/even
        if d_model % 2 != 0:
             # Slice div_term to match
             div_term = div_term[:(d_model // 2) + 1]

        pe[:, 0::2] = torch.sin(position * div_term)[:max_len, : (d_model+1)//2]
        pe[:, 1::2] = torch.cos(position * div_term)[:max_len, : d_model//2]

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # This method is not used directly in our modified forward logic above, 
        # but kept for compatibility.
        # It slices the first dimension (max_len).
        return self.pe[:, : x.size(1)]
    
class CrossChannelPatching(nn.Module):
    def __init__(self, configs):
        super().__init__()
        patch_len = configs.patch_len
        stride = configs.patch_len
        self.tokenConv = nn.Conv2d(
            in_channels=1,
            out_channels=configs.d_model,
            kernel_size=(configs.enc_in, patch_len),
            stride=(1, stride),
            padding=0,
            padding_mode="circular",
            bias=False,
        )
        # Fix Padding Logic:
        # TUEG: T_total = 30 * 200 = 6000.
        # Patch Len = 200. Stride = 200.
        # Conv2d on (B, 1, C, T) with kernel (C, P).
        # Height is C (exact match). Width is P.
        # If we use ReplicationPad1d((0, stride)), we add 200 padding to the end.
        # Input width T=6000. Padded T=6200.
        # Kernel=200, Stride=200.
        # Output Width = (6200 - 200) / 200 + 1 = 30 + 1 = 31.
        # But we expect 30 patches!
        # The original TeCh code might have expected overlap or different padding?
        # If stride == kernel_size (non-overlapping), we DO NOT need padding if T is divisible.
        
        # If stride < patch_len (overlapping), we might need padding.
        # Here stride = patch_len.
        
        # Check original code logic:
        # self.padding = nn.ReplicationPad1d((0, stride))
        # It unconditionally adds stride padding.
        
        # Fix: Remove padding if exact division is expected for non-overlapping patches.
        # Or adjust padding to 0 if stride==patch_len.
        
        if stride == patch_len:
            self.padding = nn.Identity()
        else:
            self.padding = nn.ReplicationPad1d((0, stride))
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        # x: (B, C, T)
        # x = self.padding(x) # Apply padding if needed (now conditional)
        # But ReplicationPad1d expects (B, C, T) or (B, T). It pads the LAST dimension.
        # If self.padding is Identity, no change.
        
        if not isinstance(self.padding, nn.Identity):
             x = self.padding(x)
             
        x = x.unsqueeze(1) # (B, 1, C, T)
        x = self.tokenConv(x).squeeze(2).transpose(1, 2) # (B, d_model, T_patches) -> (B, T_patches, d_model)
        return x
