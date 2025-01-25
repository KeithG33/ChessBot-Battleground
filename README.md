# ChessBot Battleground

### 🚧 Under Construction 🚧

Project under development. Stay tuned!

:)

##  Summary

This repository is designed to help you train and compare model architectures, primarily in computer vision, to see how well they can extract information and perform in chess using tiny 8x8 chess board images

The dataset is available under Releases, and contains 2000 pgn files split into train and test. A pytorch dataset is provided in the library at `chessbot.data.dataset` to load the data. 

This library provides support for:

- **Dataset/Training:** Dataset and training code for models.
- **Gym Environment:** Gym environment for inference, self-play, and displaying games
- **Dueling:** Functionality for best-of-N matches between saved models
- **MCTS:** Simple implementation to give your supervised models search capability.
- **Visualization:** Watch your models in action as they play and adapt on the board.


## 📂 Dataset

Currently the dataset contains approximately **700 million positions** in **PGN format**. For each position there is a ground-truth action and result.

Highlights include:

- Super-GM games, World Championship matches, famous tournaments
- Lichess Puzzle Database
- Stockfish vs. AlphaZero
- TCEC (Top Chess Engine Championship) database
- Including some Chess960/FR/freestyle games

With a goal of reaching **1 billion positions**, the dataset is ideal for large-scale training. 

*The dataset is available for download.*  


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

- Expand the dataset to over **1 billion positions**.  
- Release **MCTS training pipelines**.  
- Add enhanced tools for training and visualization.



## 🛠 Installation & Getting Started

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-repo/chessbot-battleground.git
   cd chessbot-battleground
2. Install dependencies
    ```bash
    pip install -r requirements.txt
3. Download the dataset (link coming soon)
4. Check out the examples 
