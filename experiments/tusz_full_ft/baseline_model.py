import torch
import torch.nn as nn


class BaselineConv1D(nn.Module):
    def __init__(self, num_channels=19, num_classes=13, dropout=0.2):
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.feat = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, mask=None):
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape (B, C, N, P), got {tuple(x.shape)}")

        b, c, n, p = x.shape
        if c != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {c}")

        xt = x.reshape(b, c, n * p)
        ft = self.feat(xt)

        if mask is None:
            pooled = ft.mean(dim=-1)
        else:
            if mask.ndim != 2 or mask.shape[0] != b or mask.shape[1] != n:
                raise ValueError(f"Expected mask with shape (B, N), got {tuple(mask.shape)}")
            valid = (~mask).to(dtype=ft.dtype, device=ft.device)
            valid_t = valid.unsqueeze(-1).expand(-1, -1, p).reshape(b, n * p)
            valid_t = valid_t.unsqueeze(1)
            ft_masked = ft * valid_t
            denom = valid_t.sum(dim=-1).clamp(min=1.0)
            pooled = ft_masked.sum(dim=-1) / denom

        return self.head(pooled)

