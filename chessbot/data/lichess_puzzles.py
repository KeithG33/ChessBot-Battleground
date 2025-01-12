"""
Used to generate PGN files from the download Lichess puzzle database in CSV format.
"""
import pandas as pd
import chess
import os

def convert_csv_to_pgn(input_csv_path, output_dir, chunk_size=1000):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    def parse_moves(moves_str, board):
        moves = moves_str.split(' ')
        san_moves = []
        move_number = int(board.fullmove_number)
        for move in moves:
            uci_move = chess.Move.from_uci(move)
            if uci_move in board.legal_moves:
                san_move = board.san(uci_move)
                board.push(uci_move)
                # Determine the proper move notation with move numbers
                formatted_move = f"{move_number}. {san_move}" if board.turn == chess.BLACK else f"{san_move}"
                san_moves.append(formatted_move)
                if board.turn == chess.WHITE:
                    move_number += 1
            else:
                return "Invalid move sequence"
        return ' '.join(san_moves) + " *"

    def generate_pgn(row):
        board = chess.Board(row['FEN'])
        # Determine the winner based on the first move and set `to_move` and `result` accordingly
        winner = 'White to move.' if board.turn == chess.BLACK else 'Black to move.'
        result = "0-1" if board.turn == chess.WHITE else "1-0"
        
        pgn_content = [
            f"[Site \"{row['GameUrl']}\"]",
            f"[Date \"{pd.to_datetime('today').strftime('%Y.%m.%d')}\"]",
            f"[ToMove \"{winner}\"]",  # Set winner as the one to move
            f"[ECO \"A00 \"]",  # Placeholder ECO, this might need to be dynamically set based on opening used
            f"[Opening \"{row['OpeningTags']}\"]",
            f"[Rating \"{row['Rating']}\"]",
            f"[Themes \"{row['Themes']}\"]",
            f"[Result \"{result}\"]",
            f"[FEN \"{row['FEN']}\"]",
            f"{parse_moves(row['Moves'], board)}"
        ]
        return '\n'.join(pgn_content) + "\n\n"

    # Read the CSV file in chunks
    chunk_iterator = pd.read_csv(input_csv_path, chunksize=chunk_size)
    
    for i, chunk in enumerate(chunk_iterator):
        print(f"Processing chunk {i + 1}")
        pgn_filename = os.path.join(output_dir, f"puzzles_{i + 1}.pgn")
        
        # Generate PGN content for all games in the chunk
        chunk_pgn_content = chunk.apply(generate_pgn, axis=1).str.cat()
        
        # Save the chunk's PGN content to a single file
        with open(pgn_filename, "w") as file:
            file.write(chunk_pgn_content)
