"""Launch the ChessBot app (Analysis + Play)."""
import argparse
from chessbot.models import ModelRegistry
from chessbot.app import launch


def main():
    parser = argparse.ArgumentParser(description="ChessBot App (Analysis + Play)")
    parser.add_argument("--model", type=str, default="swin_chessbot", help="Model name from registry")
    parser.add_argument("--weights", type=str, default="/home/kage/chess_workspace/ChessBot-Battleground/models/swin_chessbot/Swin-Chessbot-HFData-legalactions-b4096-lr0.0001-window4/model_best/pytorch_model.bin", help="Path to model weights")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top moves to display")
    parser.add_argument("--port", type=int, default=5001, help="Server port")
    args = parser.parse_args()

    model = ModelRegistry.load_model(args.model, device=args.device)
    model.load_weights(args.weights)

    launch(model, device=args.device, top_n=args.top_n, port=args.port)


if __name__ == "__main__":
    main()
