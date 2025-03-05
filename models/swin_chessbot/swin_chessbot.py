import torch
from torch import nn
import timm
import numpy as np
from torch.cuda.amp import GradScaler
from adversarial_gym.chess_env import ChessEnv



class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
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


class SwinTransformer(nn.Module):
    """
    Creates a ChessBot network that outputs a value and action for a given
    state/position. 
    
    Uses a Swin Transformer from Timm as the feature extractor, and feeds the ouput into
    prediction heads for action and value.
    """

    def __init__(self, device = 'cuda', base_lr = 0.0009, max_lr = 0.009):
        super().__init__()
        
        self.swin_transformer = timm.create_model(
            'swin_large_patch4_window7_224', 
            pretrained=False,
            img_size=8, 
            patch_size=1,
            window_size=2, 
            in_chans=1,
        ).to(device)

        self.action_dim = 4672
        self.device = device

        num_features = self.swin_transformer.head.in_features

        # Policy head
        self.policy_head = nn.Sequential(
            ResidualBlock(num_features, 2*num_features),
            nn.Linear(num_features, 2*num_features),
            nn.GELU(),
            nn.Linear(2*num_features, self.action_dim),
        ).to(device)
        
        # # Value head
        self.value_head = nn.Sequential(
            ResidualBlock(num_features, 2*num_features),
            nn.Linear(num_features, num_features),
            nn.GELU(),
            nn.Linear(num_features, 1),
            nn.Tanh()
        ).to(device)
        
    def forward(self, x):
        features = self.swin_transformer.forward_features(x) 
        features = features.view(features.size(0), -1)
        action_logits = self.policy_head(features)
        board_val = self.value_head(features)

        return action_logits, board_val
