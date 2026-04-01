import gym
import threading
import numpy as np
import chess
import torch
import torch.nn.functional as F

import adversarial_gym
from flask import Flask, render_template, request, jsonify
from datasets import load_dataset

from adversarial_gym.chess_env import ChessEnv
from chessbot.mcts import MonteCarloTreeSearch


def action_index_to_move(action_idx):
    """Decode action index (0-4671) to chess.Move without needing board.find_move."""
    from_sq, plane = np.unravel_index(action_idx, (64, 73))

    if plane < 64:
        return chess.Move(int(from_sq), int(plane))
    else:
        pd = plane - 64
        p, d = np.unravel_index(pd, (3, 3))
        promotion = int(p) + 2  # KNIGHT=2, BISHOP=3, ROOK=4
        from_file = chess.square_file(int(from_sq))
        from_rank = chess.square_rank(int(from_sq))
        to_file = int(d) - 1 + from_file
        to_rank = 0 if from_rank == 1 else 7
        to_sq = chess.square(to_file, to_rank)
        return chess.Move(int(from_sq), to_sq, promotion=promotion)


def _set_heuristic_castling(board):
    """Set castling rights heuristically based on king/rook positions."""
    rights = 0
    piece_map = board.piece_map()

    # White
    if piece_map.get(chess.E1) == chess.Piece(chess.KING, chess.WHITE):
        if piece_map.get(chess.H1) == chess.Piece(chess.ROOK, chess.WHITE):
            rights |= chess.BB_H1
        if piece_map.get(chess.A1) == chess.Piece(chess.ROOK, chess.WHITE):
            rights |= chess.BB_A1
    # Black
    if piece_map.get(chess.E8) == chess.Piece(chess.KING, chess.BLACK):
        if piece_map.get(chess.H8) == chess.Piece(chess.ROOK, chess.BLACK):
            rights |= chess.BB_H8
        if piece_map.get(chess.A8) == chess.Piece(chess.ROOK, chess.BLACK):
            rights |= chess.BB_A8

    board.castling_rights = rights


def reconstruct_board(state_tensor, legal_actions_tensor):
    """Reconstruct a chess.Board from canonical dataset state + legal actions.

    The canonical state has positive values = side-to-move's pieces.
    We try both white-to-move and black-to-move, checking which produces
    matching legal actions.
    """
    state = state_tensor.numpy().astype(int)
    legal_set = set(legal_actions_tensor.numpy().astype(int).tolist())

    for try_turn in [chess.WHITE, chess.BLACK]:
        board = chess.Board()
        board.clear()

        # Canonical: positive = side-to-move. If black to move, real = -canonical.
        real_state = state if try_turn == chess.WHITE else -state

        piece_map = {}
        for sq in range(64):
            row = sq // 8
            col = sq % 8
            val = int(real_state[row][col])
            if val != 0:
                color = chess.WHITE if val > 0 else chess.BLACK
                piece_map[chess.Square(sq)] = chess.Piece(abs(val), color)

        board.set_piece_map(piece_map)
        board.turn = try_turn
        _set_heuristic_castling(board)

        try:
            board_legal = {int(ChessEnv.move_to_action(m)) for m in board.legal_moves}
        except Exception:
            continue

        if board_legal == legal_set:
            return board

    # Fallback: return best guess (white to move) even if legal sets don't perfectly match
    board = chess.Board()
    board.clear()
    piece_map = {}
    for sq in range(64):
        row = sq // 8
        col = sq % 8
        val = int(state[row][col])
        if val != 0:
            color = chess.WHITE if val > 0 else chess.BLACK
            piece_map[chess.Square(sq)] = chess.Piece(abs(val), color)
    board.set_piece_map(piece_map)
    _set_heuristic_castling(board)
    return board


def run_inference(model, board, device, top_n=10):
    """Run model inference on a board position, return structured analysis."""
    state = ChessEnv.get_piece_configuration(board)
    # Canonical: negate for black's turn
    if board.turn == chess.BLACK:
        state = -state

    state_tensor = torch.as_tensor(
        state, dtype=torch.float32, device=device
    ).reshape(1, 1, 8, 8)

    with torch.inference_mode():
        policy_logits, value_out = model(state_tensor)

    logits = policy_logits.squeeze(0).cpu()
    value = float(value_out.squeeze().cpu())

    # Mask illegal moves
    legal_moves = list(board.legal_moves)
    legal_actions = [int(ChessEnv.move_to_action(m)) for m in legal_moves]
    legal_set = set(legal_actions)

    masked_logits = torch.full_like(logits, float('-inf'))
    for a in legal_actions:
        masked_logits[a] = logits[a]

    probs = F.softmax(masked_logits, dim=-1)

    # Top-N
    k = min(top_n, len(legal_actions))
    top_probs, top_indices = probs.topk(k)

    top_moves = []
    for rank, (prob, idx) in enumerate(zip(top_probs.tolist(), top_indices.tolist())):
        move = action_index_to_move(idx)
        # Get SAN - try with board, fallback to UCI
        try:
            san = board.san(move)
        except Exception:
            san = move.uci()

        from_name = chess.square_name(move.from_square)
        to_name = chess.square_name(move.to_square)

        top_moves.append({
            "rank": rank + 1,
            "uci": move.uci(),
            "san": san,
            "from_sq": from_name,
            "to_sq": to_name,
            "probability": round(prob, 4),
            "logit": round(float(logits[idx]), 3),
        })

    return {
        "fen": board.fen(),
        "turn": "w" if board.turn == chess.WHITE else "b",
        "model_eval": round(value, 4),
        "top_moves": top_moves,
        "legal_moves": _get_legal_dests(board),
    }


def _get_legal_dests(board):
    """Get legal move destinations as a dict {from_sq: [to_sq, ...]} for chessground."""
    dests = {}
    for move in board.legal_moves:
        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)
        if from_sq not in dests:
            dests[from_sq] = []
        if to_sq not in dests[from_sq]:
            dests[from_sq].append(to_sq)
    return dests


class DatasetPositionLoader:
    """Buffers positions from the HF streaming dataset for interactive loading."""

    def __init__(self, buffer_size=50):
        self.buffer_size = buffer_size
        self.buffer = []
        self.position_count = 0
        self._ds_iter = None
        self._lock = threading.Lock()
        self._init_dataset()

    def _init_dataset(self):
        print("Loading dataset...")
        ds = load_dataset(
            "KeithG33/ChessBot-Dataset",
            data_files={"train": "train/*.pgn.zst", "test": "test/*.pgn.zst"},
            split="test",
            streaming=True,
            trust_remote_code=True,
        ).with_format("torch")
        self._ds_iter = iter(ds.shuffle(buffer_size=1_000, seed=42))
        # Fill just a few positions to start fast; rest filled on demand
        self._fill_buffer(limit=3)
        print(f"Dataset ready, {len(self.buffer)} positions buffered.")

    def _fill_buffer(self, limit=None):
        """Fill buffer up to limit (or buffer_size)."""
        target = min(limit or self.buffer_size, self.buffer_size)
        while len(self.buffer) < target:
            try:
                example = next(self._ds_iter)
                self.buffer.append(example)
            except StopIteration:
                break

    def next_position(self):
        """Get next position from buffer, refill in background."""
        with self._lock:
            if not self.buffer:
                self._fill_buffer()
                if not self.buffer:
                    return None

            example = self.buffer.pop(0)
            self.position_count += 1

        # Refill in background
        threading.Thread(target=self._background_fill, daemon=True).start()

        state = example["state"]
        action = example["action"]
        best_action = example["best_action"]
        legal_actions = example["legal_actions"]
        result = example["result"]

        # Reconstruct board
        board = reconstruct_board(state, legal_actions)

        # Decode dataset move and best move
        action_idx = int(action.item())
        dataset_move = action_index_to_move(action_idx)
        try:
            dataset_move_san = board.san(dataset_move)
        except Exception:
            dataset_move_san = dataset_move.uci()

        best_action_idx = int(best_action.item())
        if best_action_idx >= 0:
            best_move = action_index_to_move(best_action_idx)
            try:
                dataset_best_san = board.san(best_move)
            except Exception:
                dataset_best_san = best_move.uci()
            dataset_best_uci = best_move.uci()
        else:
            dataset_best_san = None
            dataset_best_uci = None

        return {
            "board": board,
            "ground_truth_eval": round(float(result.item()), 2),
            "dataset_move_san": dataset_move_san,
            "dataset_move_uci": dataset_move.uci(),
            "dataset_best_move_san": dataset_best_san,
            "dataset_best_move_uci": dataset_best_uci,
            "position_index": self.position_count,
        }

    def _background_fill(self):
        with self._lock:
            self._fill_buffer()


def create_app(model, device="cuda", top_n=10, buffer_size=50):
    """Create Flask app for interactive analysis."""
    import os
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    app = Flask(__name__, template_folder=template_dir)

    model.eval()
    model = model.to(device)

    app.config["MODEL"] = model
    app.config["DEVICE"] = device
    app.config["TOP_N"] = top_n
    app.config["BOARD"] = chess.Board()
    app.config["ORIGINAL_FEN"] = None
    app.config["DATASET_INFO"] = {}
    app.config["MOVE_STACK"] = []       # list of UCI strings played from original position
    app.config["MOVE_INDEX"] = 0        # current position in the variation (0 = original)
    app.config["LOADER"] = DatasetPositionLoader(buffer_size=buffer_size)

    # Game play state
    app.config["GAME_BOARD"] = chess.Board()
    app.config["PLAYER_COLOR"] = None
    app.config["PLAYER_WINS"] = 0
    app.config["CPU_WINS"] = 0
    app.config["GAME_ACTIVE"] = False

    # Self-play state
    app.config["SP_BOARD"] = chess.Board()
    app.config["SP_WHITE_WINS"] = 0
    app.config["SP_BLACK_WINS"] = 0
    app.config["SP_DRAWS"] = 0
    app.config["SP_RUNNING"] = False
    app.config["SP_CONFIG"] = {}
    app.config["SP_GYM_ENV"] = None
    app.config["SP_MCTS"] = None
    app.config["SP_THINKING"] = False
    app.config["SP_PENDING_MOVE"] = None
    app.config["SP_MOVE_HISTORY"] = []   # SAN strings

    @app.route("/")
    def index():
        return render_template("eval.html", top_n=top_n)

    @app.route("/load_position", methods=["POST"])
    def load_position():
        loader = app.config["LOADER"]
        pos = loader.next_position()
        if pos is None:
            return jsonify({"error": "No more positions available"}), 404

        board = pos["board"]
        app.config["BOARD"] = board
        app.config["ORIGINAL_FEN"] = board.fen()
        app.config["MOVE_STACK"] = []
        app.config["MOVE_INDEX"] = 0
        app.config["DATASET_INFO"] = {
            "ground_truth_eval": pos["ground_truth_eval"],
            "dataset_move_san": pos["dataset_move_san"],
            "dataset_move_uci": pos["dataset_move_uci"],
            "dataset_best_move_san": pos["dataset_best_move_san"],
            "dataset_best_move_uci": pos["dataset_best_move_uci"],
            "position_index": pos["position_index"],
        }

        fen = board.fen()
        print(f"\n[Position #{pos['position_index']}] FEN: {fen}")
        print(f"  Lichess: https://lichess.org/analysis/{fen.replace(' ', '_')}")
        print(f"  GT eval: {pos['ground_truth_eval']}  |  Played: {pos['dataset_move_san']}  |  Best: {pos['dataset_best_move_san']}")

        analysis = run_inference(model, board, device, top_n)
        print(f"  Model eval: {analysis['model_eval']}  |  Top-1: {analysis['top_moves'][0]['san']} ({analysis['top_moves'][0]['probability']:.1%})")

        analysis.update(app.config["DATASET_INFO"])
        analysis["is_original"] = True
        analysis["move_index"] = 0
        analysis["move_stack"] = []
        return jsonify(analysis)

    @app.route("/analyze", methods=["POST"])
    def analyze():
        data = request.get_json()
        source = data.get("source")
        target = data.get("target")
        promotion = data.get("promotion", "")

        board = app.config["BOARD"]
        move_uci = source + target + promotion
        try:
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                # Try with queen promotion for pawn moves
                if promotion == "":
                    move = chess.Move.from_uci(move_uci + "q")
                if move not in board.legal_moves:
                    return jsonify({"error": "Illegal move"}), 400
        except Exception:
            return jsonify({"error": "Invalid move"}), 400

        board.push(move)
        app.config["BOARD"] = board

        # Truncate any forward moves if we went back, then append
        idx = app.config["MOVE_INDEX"]
        app.config["MOVE_STACK"] = app.config["MOVE_STACK"][:idx]
        app.config["MOVE_STACK"].append(move.uci())
        app.config["MOVE_INDEX"] = len(app.config["MOVE_STACK"])

        if board.is_game_over():
            return jsonify({
                "fen": board.fen(),
                "turn": "w" if board.turn == chess.WHITE else "b",
                "game_over": True,
                "result": board.result(),
                "is_original": False,
                "legal_moves": {},
                "top_moves": [],
                "model_eval": 0,
                "move_index": app.config["MOVE_INDEX"],
                "move_stack": app.config["MOVE_STACK"],
            })

        analysis = run_inference(model, board, device, top_n)
        analysis["is_original"] = False
        analysis["ground_truth_eval"] = None
        analysis["dataset_move_san"] = None
        analysis["dataset_move_uci"] = None
        analysis["dataset_best_move_san"] = None
        analysis["dataset_best_move_uci"] = None
        analysis["move_index"] = app.config["MOVE_INDEX"]
        analysis["move_stack"] = app.config["MOVE_STACK"]
        return jsonify(analysis)

    def _rebuild_board_at_index(index):
        """Rebuild board from original FEN + replaying moves up to index."""
        board = chess.Board(app.config["ORIGINAL_FEN"])
        for uci in app.config["MOVE_STACK"][:index]:
            board.push(chess.Move.from_uci(uci))
        app.config["BOARD"] = board
        app.config["MOVE_INDEX"] = index
        return board

    @app.route("/go_back", methods=["POST"])
    def go_back():
        idx = app.config["MOVE_INDEX"]
        if idx <= 0:
            return jsonify({"error": "Already at starting position"}), 400

        board = _rebuild_board_at_index(idx - 1)
        is_original = (app.config["MOVE_INDEX"] == 0)

        analysis = run_inference(model, board, device, top_n)
        if is_original:
            analysis.update(app.config["DATASET_INFO"])
        else:
            analysis["ground_truth_eval"] = None
            analysis["dataset_move_san"] = None
            analysis["dataset_move_uci"] = None
            analysis["dataset_best_move_san"] = None
            analysis["dataset_best_move_uci"] = None
        analysis["is_original"] = is_original
        analysis["move_index"] = app.config["MOVE_INDEX"]
        analysis["move_stack"] = app.config["MOVE_STACK"]
        return jsonify(analysis)

    @app.route("/go_forward", methods=["POST"])
    def go_forward():
        idx = app.config["MOVE_INDEX"]
        if idx >= len(app.config["MOVE_STACK"]):
            return jsonify({"error": "No more moves"}), 400

        board = _rebuild_board_at_index(idx + 1)

        if board.is_game_over():
            return jsonify({
                "fen": board.fen(),
                "turn": "w" if board.turn == chess.WHITE else "b",
                "game_over": True,
                "result": board.result(),
                "is_original": False,
                "legal_moves": {},
                "top_moves": [],
                "model_eval": 0,
                "move_index": app.config["MOVE_INDEX"],
                "move_stack": app.config["MOVE_STACK"],
            })

        analysis = run_inference(model, board, device, top_n)
        analysis["is_original"] = False
        analysis["ground_truth_eval"] = None
        analysis["dataset_move_san"] = None
        analysis["dataset_move_uci"] = None
        analysis["dataset_best_move_san"] = None
        analysis["dataset_best_move_uci"] = None
        analysis["move_index"] = app.config["MOVE_INDEX"]
        analysis["move_stack"] = app.config["MOVE_STACK"]
        return jsonify(analysis)

    @app.route("/reset_position", methods=["POST"])
    def reset_position():
        original_fen = app.config["ORIGINAL_FEN"]
        if original_fen is None:
            return jsonify({"error": "No position loaded"}), 400

        board = chess.Board(original_fen)
        app.config["BOARD"] = board
        app.config["MOVE_INDEX"] = 0

        analysis = run_inference(model, board, device, top_n)
        analysis.update(app.config["DATASET_INFO"])
        analysis["is_original"] = True
        analysis["move_index"] = 0
        analysis["move_stack"] = app.config["MOVE_STACK"]
        return jsonify(analysis)

    # ── Game Play Routes ────────────────────────────────────────

    def _make_bot_move(board, sample=False):
        """Bot picks best move using model inference. Returns the move."""
        state = ChessEnv.get_piece_configuration(board)
        if board.turn == chess.BLACK:
            state = -state
        legal_moves = list(board.legal_moves)
        action_idx, _ = model.get_action(state, legal_moves, sample=sample)
        # Resolve against actual legal moves to get the correct move object
        # (avoids decoded moves with wrong promotion/en-passant flags).
        legal_by_action = {int(ChessEnv.move_to_action(m)): m for m in legal_moves}
        if action_idx in legal_by_action:
            return legal_by_action[action_idx]
        # Fallback: return decoded move
        return action_index_to_move(action_idx)

    def _game_response(board, bot_move=None, game_over_msg=None):
        """Build standard game response JSON."""
        player_color = app.config["PLAYER_COLOR"]
        is_over = board.is_game_over()
        result = None
        winner = None

        if is_over or game_over_msg:
            if board.is_checkmate():
                # Side to move is in checkmate, so the other side won
                winning_color = "b" if board.turn == chess.WHITE else "w"
                winner = "player" if winning_color == player_color else "bot"
            elif game_over_msg == "resign":
                winner = "bot"
            else:
                winner = "draw"

            if winner == "player":
                app.config["PLAYER_WINS"] += 1
            elif winner == "bot":
                app.config["CPU_WINS"] += 1
            else:
                app.config["PLAYER_WINS"] += 0.5
                app.config["CPU_WINS"] += 0.5

            result = game_over_msg or board.result()

        # Get model eval for current position, normalized to player's perspective
        model_eval = 0.0
        if not is_over and not game_over_msg:
            inf = run_inference(model, board, device, top_n=1)
            raw_eval = inf["model_eval"]
            # run_inference returns value from side-to-move perspective.
            # Normalize to player_color perspective so eval bar is consistent.
            side_to_move = "w" if board.turn == chess.WHITE else "b"
            model_eval = raw_eval if side_to_move == player_color else -raw_eval

        # Build SAN move history
        move_history = []
        tmp = chess.Board()
        for m in board.move_stack:
            move_history.append(tmp.san(m))
            tmp.push(m)

        # Legal dests for player's pieces only
        legal_moves = {}
        if not is_over and not game_over_msg:
            for move in board.legal_moves:
                from_sq = chess.square_name(move.from_square)
                to_sq = chess.square_name(move.to_square)
                if from_sq not in legal_moves:
                    legal_moves[from_sq] = []
                if to_sq not in legal_moves[from_sq]:
                    legal_moves[from_sq].append(to_sq)

        resp = {
            "fen": board.fen(),
            "turn": "w" if board.turn == chess.WHITE else "b",
            "player_color": player_color,
            "game_over": bool(is_over or game_over_msg),
            "result": result,
            "winner": winner,
            "player_wins": app.config["PLAYER_WINS"],
            "cpu_wins": app.config["CPU_WINS"],
            "legal_moves": legal_moves,
            "move_history": move_history,
            "model_eval": model_eval,
        }

        if bot_move:
            resp["bot_move_from"] = chess.square_name(bot_move.from_square)
            resp["bot_move_to"] = chess.square_name(bot_move.to_square)
            try:
                # SAN before the move was pushed — reconstruct
                tmp2 = board.copy()
                tmp2.pop()
                resp["bot_move_san"] = tmp2.san(bot_move)
            except Exception:
                resp["bot_move_san"] = bot_move.uci()

        return resp

    @app.route("/game/new", methods=["POST"])
    def game_new():
        data = request.get_json()
        color = data.get("color", "w")
        if color not in ("w", "b"):
            color = "w"

        board = chess.Board()
        app.config["GAME_BOARD"] = board
        app.config["PLAYER_COLOR"] = color
        app.config["GAME_ACTIVE"] = True

        bot_move = None
        # If player is black, bot plays first as white
        if color == "b":
            bot_move = _make_bot_move(board)
            board.push(bot_move)

        print(f"\n[New Game] Player: {'White' if color == 'w' else 'Black'}")
        return jsonify(_game_response(board, bot_move=bot_move))

    @app.route("/game/move", methods=["POST"])
    def game_move():
        data = request.get_json()
        source = data.get("source")
        target = data.get("target")
        promotion = data.get("promotion", "")

        board = app.config["GAME_BOARD"]
        move_uci = source + target + promotion

        try:
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                if promotion == "":
                    move = chess.Move.from_uci(move_uci + "q")
                if move not in board.legal_moves:
                    return jsonify({"error": "Illegal move"}), 400
        except Exception:
            return jsonify({"error": "Invalid move"}), 400

        board.push(move)

        # Check if game is over after player's move
        if board.is_game_over():
            return jsonify(_game_response(board))

        # Bot responds
        bot_move = _make_bot_move(board)
        board.push(bot_move)

        return jsonify(_game_response(board, bot_move=bot_move))

    @app.route("/game/resign", methods=["POST"])
    def game_resign():
        board = app.config["GAME_BOARD"]
        resp = _game_response(board, game_over_msg="resign")
        return jsonify(resp)

    @app.route("/game/state", methods=["POST"])
    def game_state():
        board = app.config["GAME_BOARD"]
        if not app.config["GAME_ACTIVE"]:
            return jsonify({
                "game_active": False,
                "player_wins": app.config["PLAYER_WINS"],
                "cpu_wins": app.config["CPU_WINS"],
            })
        return jsonify({
            "game_active": True,
            **_game_response(board),
        })

    # ── Self-Play Routes ─────────────────────────────────────────

    def _get_mcts():
        if app.config["SP_MCTS"] is None:
            env = gym.make("Chess-v0")
            env.reset()
            app.config["SP_GYM_ENV"] = env
            app.config["SP_MCTS"] = MonteCarloTreeSearch(env, model)
        return app.config["SP_MCTS"], app.config["SP_GYM_ENV"]

    def _sp_move_history_san(board_at_start, moves_uci):
        """Rebuild SAN move history from start board + list of UCI strings."""
        tmp = chess.Board(board_at_start)
        sans = []
        for uci in moves_uci:
            m = chess.Move.from_uci(uci)
            sans.append(tmp.san(m))
            tmp.push(m)
        return sans

    def _sp_response(board, move_from=None, move_to=None, move_san=None):
        is_over = board.is_game_over()
        winner = None
        if is_over:
            r = board.result()
            if r == "1-0":
                winner = "white"
                app.config["SP_WHITE_WINS"] += 1
            elif r == "0-1":
                winner = "black"
                app.config["SP_BLACK_WINS"] += 1
            else:
                winner = "draw"
                app.config["SP_WHITE_WINS"] += 0.5
                app.config["SP_BLACK_WINS"] += 0.5
                app.config["SP_DRAWS"] += 1
            app.config["SP_RUNNING"] = False

        return {
            "fen": board.fen(),
            "turn": "w" if board.turn == chess.WHITE else "b",
            "move_from": move_from,
            "move_to": move_to,
            "move_san": move_san,
            "game_over": is_over,
            "winner": winner,
            "result": board.result() if is_over else None,
            "white_wins": app.config["SP_WHITE_WINS"],
            "black_wins": app.config["SP_BLACK_WINS"],
            "draws": app.config["SP_DRAWS"],
            "move_history": app.config["SP_MOVE_HISTORY"],
        }

    @app.route("/selfplay/start", methods=["POST"])
    def selfplay_start():
        data = request.get_json() or {}
        use_mcts = bool(data.get("use_mcts", False))
        num_sims = int(data.get("num_sims", 250))
        sample = bool(data.get("sample", False))

        board = chess.Board()
        app.config["SP_BOARD"] = board
        app.config["SP_RUNNING"] = True
        app.config["SP_THINKING"] = False
        app.config["SP_PENDING_MOVE"] = None
        app.config["SP_MOVE_HISTORY"] = []
        app.config["SP_CONFIG"] = {"use_mcts": use_mcts, "num_sims": num_sims, "sample": sample}

        print(f"\n[Self-Play] Started — MCTS={use_mcts} sims={num_sims} sample={sample}")
        return jsonify({
            "fen": board.fen(),
            "turn": "w",
            "white_wins": app.config["SP_WHITE_WINS"],
            "black_wins": app.config["SP_BLACK_WINS"],
            "draws": app.config["SP_DRAWS"],
            "move_history": [],
            "game_over": False,
        })

    @app.route("/selfplay/next_move", methods=["POST"])
    def selfplay_next_move():
        if not app.config["SP_RUNNING"]:
            return jsonify({"stopped": True})

        # MCTS: check if still computing
        if app.config["SP_THINKING"]:
            return jsonify({"thinking": True})

        # MCTS: pick up pending move from background thread
        pending = app.config["SP_PENDING_MOVE"]
        if pending is not None:
            app.config["SP_PENDING_MOVE"] = None
            board = app.config["SP_BOARD"]
            move = pending
            san = board.san(move)
            board.push(move)
            app.config["SP_MOVE_HISTORY"].append(san)
            return jsonify(_sp_response(board, chess.square_name(move.from_square),
                                        chess.square_name(move.to_square), san))

        board = app.config["SP_BOARD"]
        if board.is_game_over():
            return jsonify(_sp_response(board))

        cfg = app.config["SP_CONFIG"]

        if cfg.get("use_mcts"):
            mcts, env = _get_mcts()
            num_sims = cfg.get("num_sims", 250)

            def _mcts_thread():
                try:
                    # Sync env to current board position; returns canonical obs
                    obs = env.set_string_representation(board.fen())
                    state_str = env.get_string_representation()
                    action_idx, _ = mcts.search(state_str, obs, num_simulations=num_sims)
                    # Resolve to legal chess.Move
                    legal_by_action = {int(ChessEnv.move_to_action(m)): m for m in board.legal_moves}
                    move = legal_by_action.get(action_idx, action_index_to_move(action_idx))
                    app.config["SP_PENDING_MOVE"] = move
                except Exception as e:
                    print(f"[Self-Play MCTS error] {e}")
                    # Fallback to direct inference
                    move = _make_bot_move(board, sample=False)
                    app.config["SP_PENDING_MOVE"] = move
                finally:
                    app.config["SP_THINKING"] = False

            app.config["SP_THINKING"] = True
            threading.Thread(target=_mcts_thread, daemon=True).start()
            return jsonify({"thinking": True})

        else:
            move = _make_bot_move(board, sample=cfg.get("sample", False))
            san = board.san(move)
            board.push(move)
            app.config["SP_MOVE_HISTORY"].append(san)
            return jsonify(_sp_response(board, chess.square_name(move.from_square),
                                        chess.square_name(move.to_square), san))

    @app.route("/selfplay/stop", methods=["POST"])
    def selfplay_stop():
        app.config["SP_RUNNING"] = False
        app.config["SP_THINKING"] = False
        app.config["SP_PENDING_MOVE"] = None
        app.config["SP_BOARD"] = chess.Board()
        app.config["SP_MOVE_HISTORY"] = []
        return jsonify({
            "stopped": True,
            "white_wins": app.config["SP_WHITE_WINS"],
            "black_wins": app.config["SP_BLACK_WINS"],
            "draws": app.config["SP_DRAWS"],
        })

    return app


def launch(model, device="cuda", top_n=10, port=5001, buffer_size=50):
    """Launch the interactive analysis app in the browser.

    Args:
        model: A loaded BaseChessBot model instance.
        device: Torch device string.
        top_n: Number of top moves to display.
        port: Flask server port.
        buffer_size: Number of dataset positions to pre-buffer.
    """
    app = create_app(model, device=device, top_n=top_n, buffer_size=buffer_size)
    print(f"\n  ChessBot Analysis running at http://localhost:{port}\n")
    import os, subprocess
    def _open_browser():
        env = {**os.environ, 'QT_QPA_PLATFORM': 'offscreen'}
        subprocess.Popen(['xdg-open', f'http://localhost:{port}'], env=env)
    threading.Timer(1.5, _open_browser).start()
    app.run(host="0.0.0.0", port=port, debug=False)
