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
  
    app.config["PLAYER_WINS"] = 0
    app.config["CPU_WINS"] = 0

    @app.route("/")
    def index():
        env = app.config["ENV_INSTANCE"]
        return render_template(
            "index.html",
            fen=env.get_string_representation(),
            playerWins=app.config["PLAYER_WINS"],
            cpuWins=app.config["CPU_WINS"],
            history=[m.uci() for m in env.board.move_stack],
        )

    @app.route("/restart", methods=["POST"])
    def restart():
        env = app.config["ENV_INSTANCE"]
        env._reset_game()
        app.config["PLAYER_COLOR"] = None
        return jsonify({
            "fen": env.get_string_representation(),
            "playerWins": app.config["PLAYER_WINS"],
            "cpuWins": app.config["CPU_WINS"],
            "history": [m.uci() for m in env.board.move_stack],
        })

    @app.route("/resign", methods=["POST"])
    def resign():
        env = app.config["ENV_INSTANCE"]
        env._reset_game()
        app.config["CPU_WINS"] += 1
        return jsonify({
            "status": "resigned",
            "fen": env.get_string_representation(),
            "playerWins": app.config["PLAYER_WINS"],
            "cpuWins": app.config["CPU_WINS"],
            "history": [m.uci() for m in env.board.move_stack],
        })

    @app.route("/quit", methods=["POST"])
    def quit_game():
        try:
            webview.windows[0].destroy()
        except Exception:
            pass
        func = request.environ.get("werkzeug.server.shutdown")
        if func:
            func()
        return jsonify({"status": "quit"})
    
    def determine_winner(env):
        if env.board.is_checkmate():
            winning_color = "b" if env.board.turn else "w"
            if winning_color == app.config.get("PLAYER_COLOR", "w"):
                return "player"
            else:
                return "cpu"
        return "draw"
    
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
        return jsonify({
            "fen": env.get_string_representation(),
            "player_color": side,
            "playerWins": app.config["PLAYER_WINS"],
            "cpuWins": app.config["CPU_WINS"],
            "history": [m.uci() for m in env.board.move_stack],
        })
    
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
            winner = determine_winner(env)
            if winner == "player":
                app.config["PLAYER_WINS"] += 1
            elif winner == "cpu":
                app.config["CPU_WINS"] += 1
            elif winner == "draw":
                app.config["PLAYER_WINS"] += 0.5
                app.config["CPU_WINS"] += 0.5
            return jsonify({
                "status": "game over",
                "fen": env.get_string_representation(),
                "playerWins": app.config["PLAYER_WINS"],
                "cpuWins": app.config["CPU_WINS"],
                "history": [m.uci() for m in env.board.move_stack],
            })
        make_bot_move(env, bot_model)
        # Return win counters even when game is not over.
        return jsonify({
            "status": "ok",
            "fen": env.get_string_representation(),
            "playerWins": app.config["PLAYER_WINS"],
            "cpuWins": app.config["CPU_WINS"],
            "history": [m.uci() for m in env.board.move_stack],
        })
    return app
    

def make_bot_move(env, bot_model):
    """Make a bot move using the loaded model (or fallback to random)."""
    fen = env.get_string_representation()
    obs = env.set_string_representation(fen)
    legal_moves = env.board.legal_moves

    best_action, _ = bot_model.get_action(obs[0], legal_moves)
    bot_move = env.action_to_move(best_action)
    env.board.push(bot_move)
    

def play(bot_model, port=5000):
    app = create_app(bot_model)

    threading.Thread(target=lambda: app.run(port=port, debug=True, use_reloader=False),
                     daemon=True).start()
    time.sleep(1)
    webview.create_window("Chess Bot Battleground", "http://127.0.0.1:5000")
    webview.start()
