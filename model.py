"""
Violence detection model.

Architecture
────────────
1. SpatialEncoder  — EfficientNetV2-S (tf_efficientnetv2_s.in21k_ft_in1k)
     • Run TimeDistributed: reshape (B,T,C,H,W) → (B*T,C,H,W) → backbone → (B,T,D)
     • Backbone fine-tuned progressively (frozen head initially, unfrozen after N epochs)

2. TemporalTransformer  — lightweight 2-layer transformer encoder
     • Learnable positional encoding + CLS token
     • Multi-head self-attention captures inter-frame relationships
     • Takes CLS token output → classification head

Why transformer over LSTM:
     Dataset analysis showed motion magnitude is not discriminative (NV actually
     has MORE optical flow than V). What matters is semantic context: who is near
     whom, contact patterns, posture sequences. Transformers with attention can
     learn which frame pairs are informative rather than forcing a fixed-order
     hidden state like LSTM.
"""

import math
import torch
import torch.nn as nn
import timm
from config import Config


# ── Spatial encoder ───────────────────────────────────────────────────────────

class SpatialEncoder(nn.Module):
    """
    Wraps EfficientNetV2-S from timm.
    Input : (B, T, C, H, W)
    Output: (B, T, feature_dim)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.backbone_name,
            pretrained=True,
            num_classes=0,       # remove classifier head
            global_pool="avg",   # → (B, 1280) per image
        )
        self.feature_dim = self.backbone.num_features

        # Freeze backbone initially; call unfreeze_top_blocks() after warm-up.
        self._freeze_all()

    def _freeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_top_blocks(self, num_blocks: int = 3):
        """Unfreeze the last `num_blocks` blocks for fine-tuning."""
        blocks = list(self.backbone.blocks.children()) if hasattr(self.backbone, "blocks") \
            else list(self.backbone.children())
        for block in blocks[-num_blocks:]:
            for p in block.parameters():
                p.requires_grad_(True)
        # always unfreeze the final BN / conv head
        if hasattr(self.backbone, "conv_head"):
            for p in self.backbone.conv_head.parameters():
                p.requires_grad_(True)
        if hasattr(self.backbone, "bn2"):
            for p in self.backbone.bn2.parameters():
                p.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)           # (B*T, feature_dim)
        return feats.view(B, T, -1)        # (B, T, feature_dim)


# ── Positional encoding ────────────────────────────────────────────────────────

class LearnablePositionalEncoding(nn.Module):
    """Learnable 1-D positional embedding for sequence length T+1 (T frames + CLS)."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ── Temporal transformer ──────────────────────────────────────────────────────

class TemporalTransformer(nn.Module):
    """
    Lightweight transformer encoder over the frame-feature sequence.
    Input : (B, T, feature_dim)
    Output: (B, 1) logit
    """

    def __init__(self, cfg: Config, feature_dim: int):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, cfg.d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_enc = LearnablePositionalEncoding(cfg.num_frames + 1, cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.tf_ff_dim,
            dropout=cfg.tf_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-norm — more stable for shallow transformers
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.num_tf_layers,
            norm=nn.LayerNorm(cfg.d_model),
            enable_nested_tensor=False,  # pre-norm (norm_first=True) doesn't support nested
        )

        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.tf_dropout),
            nn.Linear(cfg.d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, feature_dim)
        x = self.input_proj(x)                    # (B, T, d_model)

        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)            # (B, T+1, d_model)
        x = self.pos_enc(x)

        x = self.transformer(x)                   # (B, T+1, d_model)
        cls_out = x[:, 0]                         # (B, d_model)
        return self.head(cls_out).squeeze(-1)     # (B,)  — raw logit


# ── Full model ────────────────────────────────────────────────────────────────

class ViolenceDetector(nn.Module):
    """
    End-to-end model: YOLO-cropped frames → SpatialEncoder → TemporalTransformer → score.
    Input : (B, T, C, H, W)  — pre-cropped by YOLO pipeline
    Output: (B,) raw logits  (apply sigmoid for probability)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.spatial = SpatialEncoder(cfg)
        self.temporal = TemporalTransformer(cfg, self.spatial.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.spatial(x)      # (B, T, feature_dim)
        return self.temporal(feats)  # (B,)

    def unfreeze_backbone(self, num_blocks: int = 3):
        self.spatial.unfreeze_top_blocks(num_blocks)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns Violence probability in [0, 1]."""
        self.eval()
        return torch.sigmoid(self(x))


# ── Parameter group helper ────────────────────────────────────────────────────

def get_param_groups(model: ViolenceDetector, cfg: Config) -> list:
    """
    Two-group learning rate: backbone (when unfrozen) uses a 10× lower LR to
    prevent catastrophic forgetting of ImageNet-21k features.
    """
    backbone_params = [p for p in model.spatial.parameters() if p.requires_grad]
    head_params = list(model.temporal.parameters())

    groups = [{"params": head_params, "lr": cfg.lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": cfg.lr * cfg.backbone_lr_scale})
    return groups
