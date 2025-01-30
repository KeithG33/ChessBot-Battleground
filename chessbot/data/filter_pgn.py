"""For parsing games in lumbras database (games not puzzles. See separate file for puzzles)"""

import os
import chess
import chess.pgn

SPECIAL_CHARS = ['?', '??', '', ' ']


def condition_met(headers, rating_threshold=2600):
    """ Return True if the game should be saved, False otherwise."""

    source = headers.get("Source", "")
    if source == "LichessEliteDatabase":
        # Rapid/Classical only - might help filter high lichess ELOs 
        time_control = headers.get("Event")
        if "Blitz" in time_control:
            return False

    header_w = headers.get("WhiteElo", 0)
    header_b = headers.get("BlackElo", 0)

    if header_w in SPECIAL_CHARS:
        header_w = 0
    if header_b in SPECIAL_CHARS:
        header_b = 0
    
    white_elo = int(header_w)
    black_elo = int(header_b)

    # If both ELO are in pgn headers then be strict
    if white_elo and black_elo:
        rating_cond = (white_elo > rating_threshold and black_elo > rating_threshold)

    # If only one ELO is in pgn headers then be lenient
    elif white_elo or black_elo:
        rating_cond = (white_elo > rating_threshold or black_elo > rating_threshold)
    
    else:
        rating_cond = False

    return rating_cond


def save_filtered_games(input_pgn_path, output_pgn_path):
    # Track games that meet the conditions
    game_offsets = []

    print(f"Processing {input_pgn_path}")
    with open(input_pgn_path, "r", encoding="ISO-8859-1") as pgn:
        while True:
            offset = pgn.tell()
            headers = chess.pgn.read_headers(pgn)

            if headers is None:
                break

            if condition_met(headers):
                game_offsets.append(offset)
            
    # Read and write games that meet conditions
    with open(input_pgn_path, "r", encoding="ISO-8859-1") as pgn:
        with open(output_pgn_path, "w", encoding="utf-8") as output_pgn:
            exporter = chess.pgn.FileExporter(output_pgn)
            for offset in game_offsets:
                pgn.seek(offset)  # Move to the game's offset
                game = chess.pgn.read_game(pgn)  # Read the full game
                game.accept(exporter)  # Save it
    
    print(f"DONE: Saved {len(game_offsets)} games to {output_pgn_path}")


def filter_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.scandir(input_dir):
        if not file.name.endswith(".pgn"):
            continue

        name = os.path.splitext(file.name)[0]
        input_pgn_path = os.path.join(input_dir, file.name)
        output_pgn_path = os.path.join(output_dir, f"{name}-2600+.pgn")
        save_filtered_games(input_pgn_path, output_pgn_path)

    
if __name__ == "__main__":
    input_dir = "/home/kage/chess_workspace/PGN_dataset/pgn-data/GM-games/Lumbras-Gigabase"
    output_dir = "/home/kage/chess_workspace/PGN_dataset/pgn-data/GM-games/Lumbras-Gigabase-2600+"

    filter_files(input_dir, output_dir)

