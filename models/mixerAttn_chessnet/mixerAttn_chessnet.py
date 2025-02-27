import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from adversarial_gym.chess_env import ChessEnv

from timm.layers import DropPath


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.norm(x)
        out = self.linear1(out) # (B, N, C) -> (B, N, Hidden)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.linear2(out) # (B, N, Hidden) -> (B, N, C)
        out = out + x
        return out


class StereoScaling(nn.Module):
    """ Learnable stereographic projection for each seq in attention map"""
    def __init__(self, seq_len: int):
        super().__init__()
        # Create the learnable x0 for stereographic projection
        self.x0 = nn.Parameter(torch.empty(1, 1, seq_len, 1))
        torch.nn.init.trunc_normal_(self.x0, std=0.02)

    def apply_stereo_scaling(self, x, x0, eps=1e-5):
        # Apply inverse stereographic projection to learnably scaled sphere
        s = (1 + x0) / (1 - x0 + eps)
        x_proj = 2 * x / (1 + s**2)
        return x_proj

    def forward( self, attn: torch.Tensor):
        attn = self.apply_stereo_scaling(attn, self.x0)
        return attn

 
class AttentionBlock(nn.Module):
    """ Multi-head self-attention module with optional 1D or 2D relative position bias.
     
    Using timm Swin Transformer implementation as a reference for the 2D relative position bias. The
    1D relative position bias is used in the token attention of the mixer block.

    Also has a learnable stereographic projection as the scale in the attention calculation, mostly for fun,
    but it also works well.
    """
    def __init__(
            self,
            seq_len, 
            embed_dim, 
            num_heads, 
            dropout=0.0, 
            use_2d_relative_position=True,
            expansion_ratio=1.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = embed_dim
        self.seq_len = seq_len
        self.use_2d_relative_position = use_2d_relative_position
        self.expanded_dim = int(embed_dim * expansion_ratio)
        self.head_dim = self.expanded_dim // num_heads
        assert self.head_dim * num_heads == self.expanded_dim, "expanded_dim must be divisible by num_heads"
        
        self.scale = StereoScaling(self.seq_len)
    
        self.qkv_proj = nn.Linear(embed_dim, 3*self.expanded_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(self.expanded_dim, embed_dim)
    
        if self.use_2d_relative_position:
            self.h, self.w = self.compute_grid_dimensions(seq_len)
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * self.h - 1) * (2 * self.w - 1), num_heads)
            )
            self.register_buffer("relative_position_index", self.get_2d_relative_position_index(self.h, self.w))
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        else:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(2 * seq_len - 1, num_heads)
            )
            self.register_buffer("relative_position_index", self.get_1d_relative_position_index(seq_len))  
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def compute_grid_dimensions(self, n):
        """
        Compute grid dimensions (h, w) for 2D relative position bias. In our case, this will be the
        height and width of the chess board (8x8).
        """
        root = int(math.sqrt(n))
        for i in range(root, 0, -1):
            if n % i == 0:
                return (i, n // i)

    def get_2d_relative_position_index(self, h, w):
        """ Create pairwise relative position index for 2D grid."""

        coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij'))  # 2, h, w
        coords_flatten = coords.reshape(2, -1)  # 2, h*w
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, h*w, h*w
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # h*w, h*w, 2
        relative_coords[:, :, 0] += h - 1  # Shift to start from 0
        relative_coords[:, :, 1] += w - 1
        relative_coords[:, :, 0] *= 2 * w - 1 
        relative_position_index = relative_coords.sum(-1)  # Shape: (h*w, h*w)
        return relative_position_index  # h*w, h*w

    def get_1d_relative_position_index(self, seq_len):
        # Compute relative position indices for 1D sequences
        coords = torch.arange(seq_len)
        relative_coords = coords[None, :] - coords[:, None]  # seq_len, seq_len
        relative_coords += seq_len - 1  # Shift to start from 0
        return relative_coords  # seq_len, seq_len

    def _get_rel_pos_bias(self):
        """Retrieve relative position bias based on precomputed indices for the attention scores."""
        # Retrieve and reshape the relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1).long()
        ]
        relative_position_bias = relative_position_bias.view(self.seq_len, self.seq_len, -1)  # seq_len, seq_len, num_heads
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # num_heads, seq_len, seq_len
        return relative_position_bias.unsqueeze(0)  # 1, num_heads, seq_len, seq_len

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) 

        relative_position_bias = self._get_rel_pos_bias() # 1, H, N, N
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=relative_position_bias,
        )
        attn = self.scale(attn)
        x = attn.transpose(1, 2).reshape(B, N, self.expanded_dim) 
        x = self.proj(x) 
        x = self.dropout(x)

        return x


class MixerBlock(nn.Module):
    def __init__(self, piece_embed_dim, num_heads=16, dropout=0., drop_path=0.):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.channel_mixing_norm = nn.LayerNorm(piece_embed_dim)
        self.channel_mixing_attn = AttentionBlock(
            64,
            piece_embed_dim, 
            num_heads, 
            dropout=dropout, 
            use_2d_relative_position=True,
            expansion_ratio=2.0,
        )
        self.token_mixing_norm = nn.LayerNorm(piece_embed_dim)
        self.token_mixing_attn = AttentionBlock(
            piece_embed_dim, 
            64, 
            16, 
            dropout=dropout, 
            use_2d_relative_position=False,
            expansion_ratio=2.0,
        )

        total_features = 64 * piece_embed_dim
        self.out_mlp = ResidualBlock(total_features, int(2*total_features), dropout=dropout)

    def forward(self, x):
        # x shape: (B, 64, piece_embed)
        x = x + self.drop_path(self.token_mixing_attn(self.token_mixing_norm(x).transpose(1,2)).transpose(1,2))
        x = x + self.drop_path(self.channel_mixing_attn(self.channel_mixing_norm(x)))
        x = self.out_mlp(x.view(x.size(0), -1)).view(x.size(0), 64, -1)
        return x


class ChessTransformer(nn.Module):
    """
    Creates a ChessNetwork that outputs a value and action for a given
    state/position using a Mixer-style network.
    
    The network processes the input state using an embedding for pieces and Mixer blocks,
    then feeds the output into separate prediction heads for policy, value, and critic outputs.
    """

    def __init__(self, device='cuda', dropout=0.0):
        super().__init__()
        self.device = device
        self.action_dim = 4672
        action_embed_dim = 512
        piece_embed_dim=24
        self.action_embed = nn.Embedding(self.action_dim, action_embed_dim).to(device)

        num_features = 64  * piece_embed_dim

        # Embedding layer for pieces (integers ranging between -6 and 6)
        self.piece_embedding = nn.Embedding(num_embeddings=13, embedding_dim=piece_embed_dim).to(device)
        self.pos_encoding = nn.Parameter(torch.randn(1, 64, piece_embed_dim))
        torch.nn.init.kaiming_normal_(self.pos_encoding, mode='fan_out', nonlinearity='relu')

        # Mixer blocks
        # params: [num_heads, drop_path]
        # TODO: not hardcode, make stages configurable
        params_config = [(6, 0.05)] * 4 + [(8, 0.1)] * 4 + [(12, 0.15)] * 8 + [(24, 0.2)] * 4
        
        self.mixer_layers = nn.Sequential(*[
            MixerBlock(piece_embed_dim, num_heads=params[0], drop_path=params[1])
            for params in params_config
        ]).to(device)

        print(f"Num mixer params: {sum(p.numel() for p in self.mixer_layers.parameters())}")

        # Policy head
        self.policy_head = nn.Sequential(
            ResidualBlock(num_features, int(2*num_features)),
            nn.LayerNorm(num_features),
            nn.Linear(num_features, num_features),
            nn.GELU(),
            nn.Linear(num_features, self.action_dim),
        ).to(device)
        
        # Value head
        self.value_head = nn.Sequential(
            ResidualBlock(num_features, int(2*num_features)),
            nn.LayerNorm(num_features),
            nn.Linear(num_features, num_features),
            nn.GELU(),
            nn.Linear(num_features, 1),
            nn.Tanh()
        ).to(device)

        self.value_loss = nn.MSELoss()
        self.policy_loss = nn.CrossEntropyLoss()

    def embed_state(self, x):
        x = x.view(x.size(0), -1)  # (B, 1, 8, 8) -> (B, 64)
        x = self.piece_embedding(x + 6)  # (B, 64) -> (B, 64, piece_embed_dim)
        x = x + self.pos_encoding.expand(x.size(0), -1, -1)
        return x

    def forward(self, x):
        features = self.embed_state(x.long())  # -> (B, 64, piece_embed_dim)
        features = self.mixer_layers(features)  # -> (B, 64, piece_embed_dim)
        features = features.view(features.size(0), -1)

        action_logits = self.policy_head(features)
        board_val = self.value_head(features)
        return action_logits, board_val
