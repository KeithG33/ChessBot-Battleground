import torch
from torch import nn
import timm

from chessbot.models import BaseChessBot, ModelRegistry


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        out = self.linear1(self.norm1(x))
        out = self.linear2(self.gelu(out))
        out = self.norm2(out + x)
        return out


@ModelRegistry.register("swin_future_chessbot")
class SwinFutureChessBot(BaseChessBot):
    """Swin-based chess model that predicts a future board state and uses it as
    additional context for policy and value heads."""

    def __init__(self, device: str = "cuda", piece_embed_dim: int = 32):
        super().__init__()
        self.device = device
        self.action_dim = 4672

        self.swin_transformer = timm.create_model(
            "swin_large_patch4_window7_224",
            pretrained=False,
            img_size=8,
            patch_size=1,
            window_size=2,
            in_chans=1,
        ).to(device)

        num_features = self.swin_transformer.head.in_features
        self.piece_embedding = nn.Embedding(13, piece_embed_dim).to(device)

        self.future_head = nn.Sequential(
            ResidualBlock(num_features, 2 * num_features),
            nn.Linear(num_features, 64 * 13),  # logits for 13 piece types per square
        ).to(device)

        self.future_proj = nn.Linear(64 * piece_embed_dim, num_features).to(device)

        combined_dim = num_features * 2

        self.policy_head = nn.Sequential(
            ResidualBlock(combined_dim, 2 * combined_dim),
            nn.Linear(combined_dim, 2 * combined_dim),
            nn.GELU(),
            nn.Linear(2 * combined_dim, self.action_dim),
        ).to(device)

        self.value_head = nn.Sequential(
            ResidualBlock(combined_dim, 2 * combined_dim),
            nn.Linear(combined_dim, combined_dim),
            nn.GELU(),
            nn.Linear(combined_dim, 1),
            nn.Tanh(),
        ).to(device)

    def _future_feature(self, features: torch.Tensor) -> torch.Tensor:
        """Return embedding of the predicted future board state."""
        B = features.size(0)
        future_logits = self.future_head(features)
        future_logits = future_logits.view(B, 64, 13)
        probs = torch.softmax(future_logits, dim=-1)
        future_embed = probs @ self.piece_embedding.weight
        future_embed = future_embed.view(B, -1)
        return self.future_proj(future_embed)

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        features = self.swin_transformer.forward_features(x)
        features = features.view(B, -1)

        future_feat = self._future_feature(features)

        combined = torch.cat([features, future_feat], dim=-1)

        action_logits = self.policy_head(combined)
        board_val = self.value_head(combined)
        return action_logits, board_val
