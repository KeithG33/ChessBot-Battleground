from chessbot.inference import selfplay, run_match
from simple_chessbot import SimpleChessBot

model = SimpleChessBot(hidden_dim=512)
# model.load_state_dict(torch.load('pytorch_model.bin'))

outcome = selfplay(
  model, 
  search=True,   # Use MCTS search
  num_sims=250,  # How many simulations to run
  visualize=True # Display the game
)

scores = run_match(
  model,  # player1 model
  model,  # player2 model
  best_of=7,      # Best-of 
  search=False,   # Use MCTS search
  num_sims=250,   # Num sims if searching
  visualize=True, # Display the game
  sample=False,   # Sample or select best from policy distribution
)