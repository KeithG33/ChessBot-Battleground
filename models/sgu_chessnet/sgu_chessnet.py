import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from adversarial_gym.chess_env import ChessEnv


from chessbot.models.base import BaseChessModel


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

 
class TinyAttentionBlock(nn.Module):
    """ Multi-head self-attention module with 1D relative position bias and learnable scale parameter.
    """
    def __init__(
            self,
            seq_len, 
            embed_dim, 
            expansion_ratio=1.0,
            dim_out=None
    ):
        super().__init__()
        self.dim = embed_dim
        self.seq_len = seq_len
        self.expanded_dim = int(embed_dim * expansion_ratio)
        dim_out = dim_out or embed_dim
        
        self.scale = nn.Parameter(torch.tensor(1 / (self.dim ** 0.5)))
        
        self.qkv_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3*self.expanded_dim)
        )
        
        self.proj = nn.Sequential(
            nn.Linear(self.expanded_dim, dim_out),
            nn.LayerNorm(dim_out)
        )
    
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * seq_len - 1)
        )
        self.register_buffer("relative_position_index", self.get_1d_relative_position_index(seq_len))  
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

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
        relative_position_bias = relative_position_bias.view(self.seq_len, self.seq_len)  # seq_len, seq_len,
        relative_position_bias = relative_position_bias.contiguous()  #  seq_len, seq_len
        return relative_position_bias

    def forward(self, x):
        B, N, C = x.shape
        
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)  # Split along the last dimension

        relative_position_bias = self._get_rel_pos_bias() # 1, H, N, N

        attn = torch.matmul(q, k.transpose(-2, -1))  # B, H, N, N
        attn = attn + relative_position_bias

        attn = F.softmax(attn, dim=-1)
        attn = attn * self.scale

        x = torch.matmul(attn, v)
        x = self.proj(x) 
        return x


class SpatialGatingUnit(nn.Module):
    """ My simple naive implementation of SGU from the pseudo-code in the paper. """
    def __init__(self, dim, dim_seq, expansion=4):
        super().__init__()  
        dim_sgu = expansion * dim
        dim_attn = dim_sgu // 2

        self.proj_in = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_sgu),
            nn.GELU()
        )
        self.proj_spatial = nn.Sequential(
            nn.LayerNorm(dim_seq),
            nn.Linear(dim_seq, dim_seq),
            nn.GELU(),
        )

        self.proj_out = nn.Sequential(
            nn.LayerNorm(dim_sgu//2),
            nn.Linear(dim_sgu//2, dim),
        )

        self.attn = TinyAttentionBlock(dim_seq, dim, expansion_ratio=expansion, dim_out=dim_attn)

        # set the spatial projection bias to 1
        nn.init.constant_(self.proj_spatial[1].bias, 1.0)

    def forward(self, x):
        a = self.attn(x)
        x = self.proj_in(x)
        u, v = x.chunk(2, dim=-1)
        v = self.proj_spatial(v.transpose(1,2)).transpose(1,2)

        out = u * (v + a)
        out = self.proj_out(out)
        return out
    

class SGUBlock(nn.Module):
    def __init__(self, piece_embed_dim, dropout=0.):
        super().__init__()
        total_features = 64 * piece_embed_dim
        self.mlp = ResidualBlock(total_features, int(1.1*total_features), dropout=dropout)
        self.spatial_gating = SpatialGatingUnit(piece_embed_dim, 64, expansion=4)
        
    def forward(self, x):
        # x shape: (B, 64, piece_embed)
        x = x + self.spatial_gating(x)
        x = self.mlp(x.view(x.size(0), -1)).view(x.size(0), 64, -1)
        return x


class SpatialGatingChessNet(BaseChessModel):
    """ Spatial Gating Units (SGU) paired with residual MLP blocks for global feature processing.
    """

    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.action_dim = 4672
        piece_embed_dim = 32
        model_dim = 32

        num_features = 64 * model_dim

        # Embedding layer for pieces (integers ranging between -6 and 6)
        self.piece_embedding = nn.Embedding(num_embeddings=13, embedding_dim=piece_embed_dim).to(device)
        self.pos_encoding = nn.Parameter(torch.randn(1, 64, piece_embed_dim))
        torch.nn.init.kaiming_normal_(self.pos_encoding, mode='fan_out', nonlinearity='relu')
        self.proj_in = nn.Linear(piece_embed_dim, model_dim)

        params_config = [(0.1,)] * 4 + [(0.15,)] * 4 + [(0.2,)] * 12  + [(0.2,)] * 4

        self.blocks = nn.Sequential(*[
            SGUBlock(model_dim, dropout=params[0])
            for params in params_config
        ]).to(device)

        print(f"Num sgu block params: {sum(p.numel() for p in self.blocks.parameters())}")

        # Policy head
        self.policy_head = nn.Sequential(
            ResidualBlock(num_features, int(2*num_features)),
            nn.LayerNorm(num_features),
            nn.Linear(num_features, int(1.5*num_features)),
            nn.GELU(),
            nn.Linear(int(1.5*num_features), self.action_dim),
        ).to(device)
        
        # Value head
        self.value_head = nn.Sequential(
            ResidualBlock(num_features, int(2*num_features)),
            nn.LayerNorm(num_features),
            nn.Linear(num_features, int(1.5*num_features)),
            nn.GELU(),
            nn.Linear(int(1.5*num_features), 1),
            nn.Tanh()
        ).to(device)

        self.value_loss = nn.MSELoss()
        self.policy_loss = nn.CrossEntropyLoss()
    
    def embed_state(self, x):
        x = x.view(x.size(0), -1).long()  # (B, 1, 8, 8) -> (B, 64)
        x = self.piece_embedding(x + 6)  # (B, 64) -> (B, 64, piece_embed_dim)
        x = x + self.pos_encoding.expand(x.size(0), -1, -1)
        x = self.proj_in(x)
        return x

    def forward(self, x):
        features = self.embed_state(x)  # Shape: (B, 64, piece_embed_dim)
        features = self.blocks(features)  # Shape: (B, piece_embed_dim, 64)
        features = features.view(features.size(0), -1)

        action_logits = self.policy_head(features)
        board_val = self.value_head(features)

        return action_logits, board_val
