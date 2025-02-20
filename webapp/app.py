# import random
# import threading
# import time

# from flask import Flask, render_template, request, jsonify
# import chess
# import webview

# from chessbot.models import ModelRegistry
# from adversarial_gym.chess_env import ChessEnv

# app = Flask(__name__)

# # Global game state and player color.
# game = chess.Board()
# env = ChessEnv()
# player_color = "w"  # default; will be updated via /set_side

# # Load model from registry
# model_dir = '/home/kage/chess_workspace/ChessBot-Battleground/examples/example_model'
# model_name = 'simple_chessnet'
# bot_model = ModelRegistry.load_model_from_directory(model_name, model_dir)

# def make_bot_move():
#     """Make a bot move using the loaded model (or fallback to random)."""
#     global game
#     fen = game.fen()
#     obs = env.set_string_representation(fen)
#     legal_moves = env.board.legal_moves

#     best_action, _ = bot_model.get_action(obs[0], legal_moves)
#     bot_move = env.action_to_move(best_action)
#     game.push(bot_move)

# @app.route("/")
# def index():
#     # Render the page with the starting FEN.
#     # (We no longer pass player_color here because the client chooses it.)
#     return render_template("index.html", fen=game.fen())

# @app.route("/restart", methods=["POST"])
# def restart():
#     global game, player_color
#     game = chess.Board()  # reinitialize the game
#     player_color = None   # clear the previously chosen side
#     return jsonify({"fen": game.fen()})

# @app.route("/set_side", methods=["POST"])
# def set_side():
#     global player_color, game
#     data = request.get_json()
#     side = data.get("side", "w")
#     if side not in ["w", "b"]:
#         side = "w"
#     player_color = side
#     # If the player is Black, let the bot (playing White) make the first move.
#     if player_color == "b":
#         make_bot_move()
#     return jsonify({"fen": game.fen(), "player_color": player_color})

# @app.route("/move", methods=["POST"])
# def move():
#     global game
#     data = request.get_json()
#     source = data.get("source")
#     target = data.get("target")
#     promotion = data.get("promotion", "")
#     move_uci = source + target + promotion
#     move_obj = chess.Move.from_uci(move_uci)
    
#     # Process the player's move.
#     if move_obj not in game.legal_moves:
#         return jsonify({"status": "illegal move", "fen": game.fen()}), 400
    
#     game.push(move_obj)
    
#     if game.is_game_over():
#         return jsonify({"status": "game over", "fen": game.fen()})
    
#     make_bot_move()
    
#     return jsonify({"status": "ok", "fen": game.fen()})

# def start_flask():
#     app.run(port=5000, debug=True, use_reloader=False)

# if __name__ == "__main__":
#     threading.Thread(target=start_flask, daemon=True).start()
#     # Wait briefly to ensure the server is running.
#     time.sleep(1)
#     webview.create_window("Chess App", "http://127.0.0.1:5000")
#     webview.start()
import random
import threading
import time
import argparse
import sys
import chess
import webview
import torch

from flask import Flask, render_template, request, jsonify

from chessbot.models import ModelRegistry
from adversarial_gym.chess_env import ChessEnv


def create_app(bot_model=None):
    app = Flask(__name__)
    # Create a global ChessEnv instance and initial board.
    app.config["ENV_INSTANCE"] = ChessEnv()
    app.config["PLAYER_COLOR"] = None

    @app.route("/")
    def index():
        env = app.config["ENV_INSTANCE"]
        return render_template("index.html", fen=env.get_string_representation())

    @app.route("/restart", methods=["POST"])
    def restart():
        env = app.config["ENV_INSTANCE"]
        env._reset_game()
        app.config["PLAYER_COLOR"] = None
        return jsonify({"fen": env.get_string_representation()})

    @app.route("/set_side", methods=["POST"])
    def set_side():
        env = app.config["ENV_INSTANCE"]
        data = request.get_json()
        side = data.get("side", "w")
        if side not in ["w", "b"]:
            side = "w"
        app.config["PLAYER_COLOR"] = side
        # If the user is Black, let the bot (White) make the first move.
        if side == "b":
            make_bot_move(env, bot_model)
        return jsonify({"fen": env.get_string_representation(), "player_color": side})

    @app.route("/move", methods=["POST"])
    def move():
        env = app.config["ENV_INSTANCE"]
        data = request.get_json()
        source = data.get("source")
        target = data.get("target")
        promotion = data.get("promotion", "")
        move_uci = source + target + promotion
        human_move = chess.Move.from_uci(move_uci)
        if human_move not in env.board.legal_moves:
            return jsonify({"status": "illegal move", "fen": env.get_string_representation()}), 400
        env.board.push(human_move)
        if env.board.is_game_over():
            return jsonify({"status": "game over", "fen": env.get_string_representation()})
        make_bot_move(env, bot_model)
        return jsonify({"status": "ok", "fen": env.get_string_representation()})

    return app

def make_bot_move(env, bot_model):
    """Make a bot move using the loaded model (or fallback to random)."""
    fen = env.get_string_representation()
    obs = env.set_string_representation(fen)
    legal_moves = env.board.legal_moves

    best_action, _ = bot_model.get_action(obs[0], legal_moves)
    bot_move = env.action_to_move(best_action)
    env.board.push(bot_move)
    

def main():
    parser = argparse.ArgumentParser(description="Run Chess Bot Battleground")
    parser.add_argument("--model_dir", type=str, default=None, help="Directory containing the model")
    parser.add_argument("--model_name", type=str, default=None, help="Name of the model to load")
    parser.add_argument("--model_weights", type=str, default=None, help="Path to the model weights file")
    args = parser.parse_args()

    bot_model = None
    if args.model_dir and args.model_name:
        bot_model = ModelRegistry.load_model_from_directory(args.model_name, args.model_dir)
        if args.model_weights:
            state_dict = torch.load(args.model_weights)
            bot_model.load_state_dict(state_dict).cuda()
    else:
        print(f"Please pass a model directory and name to load a model.")
        sys.exit(1)

    app = create_app(bot_model)

    threading.Thread(target=lambda: app.run(port=5000, debug=True, use_reloader=False),
                     daemon=True).start()
    time.sleep(1)
    webview.create_window("Chess Bot Battleground", "http://127.0.0.1:5000")
    webview.start()


def play(bot_model):
    app = create_app(bot_model)

    threading.Thread(target=lambda: app.run(port=5000, debug=True, use_reloader=False),
                     daemon=True).start()
    time.sleep(1)
    webview.create_window("Chess Bot Battleground", "http://127.0.0.1:5000")
    webview.start()
    

if __name__ == "__main__":
    main()
