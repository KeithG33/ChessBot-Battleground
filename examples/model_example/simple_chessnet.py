import sys
sys.path.append('/home/kage/chess_workspace/ChessBot-Battleground')

import chessbot 
from chessbot.models.base import BaseModel
from chessbot.models.registry import ModelRegistry

import torch
import torch.nn as nn

@ModelRegistry.register(name='simple_chessnet')
class SimpleChessNet(BaseModel):
    """
    A simple example model that extends the BaseModel for the ChessBot-Battleground library.
    This model uses a few linear layers to process the input state and produce action logits
    and a board value.
    """
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        piece_embed_dim = 4
        features = 64 * piece_embed_dim
        
        self.piece_embedding = nn.Embedding(num_embeddings=13, embedding_dim=piece_embed_dim).to(device)

        # Simple mlp and some prediction heads
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features, features),  # Flatten the 8x8 board and process
            nn.ReLU(),
            nn.LayerNorm(features),
            nn.Linear(features, features),
            nn.ReLU(),
        ).to(self.device)

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(piece_embed_dim * 64, self.action_dim)
        ).to(self.device)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(piece_embed_dim * 64, 1),
            nn.Tanh()  # Output a value between -1 and 1
        ).to(self.device)
    
    def board_embedding(self, x):
        """ Embed the board state."""
        x = x.view(-1, 64)
        return self.piece_embedding(x + 6)  # Shift the piece values to [0,13]

    def forward(self, x):
        """
        Process the input state and produce action logits and board value.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, 8, 8)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
            action_logits: shape (B, 4672)
            board_val: shape (B,)
        """
        x = self.board_embedding(x.long())
        features = self.layers(x)  # Reshape and pass through the network
        action_logits = self.policy_head(features)
        board_val = self.value_head(features)
        return action_logits, board_val
    
# model = SimpleChessNet().cuda()
# print(f"Num parameters: {sum(p.numel() for p in model.parameters())}")

# # Test the transformer with random inputs
# state = torch.randint(-1, 1, (32, 1, 8, 8)) * 6  # Random integer tensor between -6 and 6

# # # Forward pass
# action_logits, board_val = model(state.cuda())
