from chessbot.models import MODEL_REGISTRY
from chessbot.inference import selfplay, play_match

# Run selfplay
model1 = MODEL_REGISTRY.load_model("simple_chessbot")
outcome = selfplay(model1, search=True, visualize=False, sample=True) # 1 if white wins, -1 if black wins, 0 draw

# Play a match with two models, use MCTS
model2 = MODEL_REGISTRY.load_model("simple_chessbot")
scores = play_match(model1, model2, best_of=3, search=True, visualize=True) # Returns (score1, score2)
