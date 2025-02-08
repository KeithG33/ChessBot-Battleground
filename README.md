# ChessBot Battleground

### 🚧 Under Construction 🚧

Project under development. Stay tuned!



##  Introduction

This repository is designed to help you train and compare model architectures, primarily in computer vision, to see how well they can extract information and perform in chess with tiny 8x8 chess board images

The dataset is available under Releases, and contains 2000 pgn files split into train and test. A pytorch dataset is provided in the library at `chessbot.data.dataset` to load the data. 

This library provides support for:

- **Dataset/Training:** Dataset and training code for models.
- **Gym Environment:** Gym environment for inference, self-play, and displaying games
- **Dueling:** Functionality for best-of-N matches between saved models
- **MCTS:** Simple implementation to give your supervised models search capability.
- **Visualization:** Watch your models in action as they play and adapt on the board.


## 📂 Dataset

Currently the dataset contains approximately **700 million positions** in **PGN format**. 
The repo provides a pytorch dataset to load the data and it is mainly taken from the following sources:

- Lumbra's Database (filtered 2600+)
- Lichess Puzzle Database
- Computer Chess - TCEC Database, CCRL, Stockfish vs. AlphaZero, Kasparov vs Deep Blue
- Includes chess960/FR/freestyle
- Stockfish Data - position evaluation, puzzle solutions, best move sequences

With a goal of reaching **1 billion positions**, the dataset is ideal for large-scale training. 

*See the latest Release for the dataset .zip file*  


## 🧠 Training

A training pipeline is provided, however it was designed for my machine, so users may want to write their own training code.

- **PyTorch Dataset:** Load positions directly from PGN files with multiprocessing for fast data loading.
- **Data Distribution:** The dataset is split into **2000 PGN files**, each containing approximately **350,000 positions**.
- **Trainer:**  
   - Designed for mid-high RAM environments.  
   - Trains in **rounds** and **epochs**:  
     - Each round samples a specified number of training PGN files.  
     - Performs a set number of epochs before moving to next round.
     - 7 PGN ~= 1Gb of RAM


## 🤖 Inference & Battling

Take your models to the battleground! The library includes:

- **Adversarial Gym Environment:**  
  - Designed for **two-player games.**  
  - Visualize and analyze your bots battling it out on the board.
  - Dueling best-of-N matches between your saved models
  
- **Monte Carlo Tree Search (MCTS):**  
  - Leverage MCTS to enhance your bots' decision-making.  
  - Implement search-based strategies for stronger gameplay.  

**MCTS Training Code:** Coming soon!


## 📈 Future Plans

- Clean, deduplicate, and expand the dataset, with an epic milestone of **1 billion positions**.  
- Release **MCTS training pipelines**.  
- Add enhanced tools for training and visualization.


## 🛠 Installation

1. From source:  
   ```bash
   git clone https://github.com/KeithG33/ChessBot-Battleground.git

   cd chessbot-battleground

   pip install -r requirements.txt
   pip install . 
   ```

2. Pip 
    ```bash
    pip install chessbot
    ```

## Getting Started
**1. Downloading the dataset:**  

The dataset is provided as a downloadable .zip file with each release. Either navigate to the github release page, or use the provided download tool:
```bash
chessbot download [<tag>="latest"] [--save-path <path>] [--dataset-name <filename>]

# Ex: download latest release to cwd if pip installed, or ChessBot-Battleground/dataset if source installed
chessbot download

# Ex: download release v0.0.0 to output_dir
chessbot download v0.0.0 --output-dir /path/to/output_dir
```

By default the latest release will be downloaded into the `ChessBot-Battleground/dataset/` directory
