import torch
from chess_transformer.chess_transformer.chess_network_sgu import ChessTransformer
from chessbot.models import MODEL_REGISTRY
from chessbot.inference.evaluate import evaluate_model
from chessbot.models import align_state_dict



MODEL_PATH = '/home/kage/chess_workspace/chess24-sgu-nocrit-R8.1-chkpt.pt'

model = MODEL_REGISTRY.load_model('sgu_chessbot')
model.load_state_dict(torch.load(MODEL_PATH))
model.to('cuda')
model.eval()

# Evaluate the model
batch_size = 3072
num_processes = 0
pgn_dir = '/home/kage/chess_workspace/ChessBot-Battleground/dataset/test-0.0.0/'

evaluate_model(
    model, 
    pgn_dir, 
    batch_size, 
    num_processes, 
    device='cuda')
