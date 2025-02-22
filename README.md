
<!-- Banner Start -->
<div align="center">
<img src="assets/chessbot-banner.png" style="width: 45%; height: auto;">  

# ChessBot Battleground

**Chess AI Training & Battleground Platform**

[**Getting Started (Installation)**](#getting-started) • [**Dataset**](#-dataset) • [**Training**](#-training) • [**Inference & Battling**](#-inference--battling) • [**Models**](#models)  

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FKeithG33%2FChessBot-Battleground&count_bg=%23A7AFB3&title_bg=%23170532&icon=&icon_color=%230019EE&title=views&edge_flat=false)](https://hits.seeyoufarm.com)

</div>
<!-- Banner End -->

##  Introduction

This repository contains a gigantic curated chess dataset meant for machine learning, along with the supporting code to train, infer, and display games. Just design a model and you can take advantage of any of the available features. This library provides support for:

- **Dataset/Training:** PyTorch dataset and training code
- **Evaluation**: Compare performance on the dataset. *Leaderboard coming soon*
- **Visualization:** Watch your models in action as they play and adapt on the board.
- **MCTS:** Simple implementation to give your supervised models search capability. *Training coming soon*.
- **Game App:**  Play a game against your model with the `chessbot play` tool. Can it beat you?

See a model in action below:

<div align="center"  id="chess-battle-gif">
  <img src="assets/chessSGU-R8.1-selfplay.gif" style="width: 35%; height: auto;">
  <p><em>Self-play after a few days of training a 300M parameter network from scratch on an RTX 3090.</em></p>
</div>


## 📂 Dataset
*[Releases available for download here](https://github.com/KeithG33/ChessBot-Battleground/releases)*  

Currently the dataset contains approximately **700 million positions** in **PGN format**, split across 1000 files. Huge credits to the following main sources:

- Lumbra's Database (filtered 2600+)
- Lichess Puzzle Database
- Computer Chess: TCEC Database, CCRL
<!-- - (*coming soon*) Stockfish Data: position evaluation, puzzle solutions, best-move sequences   -->
  
      
The PyTorch `ChessDataset` is provided in `chessbot/data/dataset.py` to load the data for training, which has the following format:

```python
import chessbot.data.ChessDataset

pgn_files   = 'path/to/pgn_files' # File, directory, or list of files
num_threads = 8

dataset     = ChessDataset(pgn_files, num_threads=num_threads)
dataloader  = DataLoader(dataset, batch_size=bsz)
batch       = next(iter(dataloader))

# Check shapes
states  = batch[0]  # (B, 8, 8),
actions = batch[1]  # (B, 4672)
results = batch[2]  # (B,)
```

If you have your own pgn files and want to make your own dataset, the PyTorch dataset should work with those as well. The moves and results are loaded from each game in the PGN file
## 🧠 Training

A `ChessTrainer` class can be used to train ChessBot models. The trainer splits the data loading and training into rounds and epochs. Each round will sample a new subset of `cfg.dataset.size_train` files, and then perform epochs on this subset.   


**ChessTrainer**
- HuggingFace's `accelerate` for easy access to mixed precision, compilation, gradient clipping, gradient accumulation, etc.
- Warmup LR scheduling and decay (linear or cosine)
- Validation frequency in iterations
- Optional logging to wandb 


Here's a somewhat realistic example of using it:

```python
from chessbot.config import load_default_cfg()
from chessbot.train import ChessTrainer

# Get default cfg and do some basic setup
cfg = load_default_cfg() # get default cfg

cfg.train.rounds = 1 # num times to sample a dataset
cfg.train.epochs = 25 # num epochs on sampled dataset
cfg.train.batch_size = 128
cfg.train.lr = 0.001
cfg.train.output_dir = 'output/'

cfg.dataset.data_path = 'ChessBot-Battleground/dataset/'
cfg.dataset.size_train = 25 # num files to sample for train set
cfg.dataset.size_test = 5 # num files to sample for test set

model = YourChessModel()

trainer = ChessTrainer(cfg, model)
trainer.train()
```

See [`chessbot/train/config.yaml`](chessbot/train/config.yaml) for a list and description of the available options, and the next section for a brief description of models.


## 🤖 Models

Design your model and see how it does! To take advantage of the training and inference code, models should subclass the `BaseChessModel` class and follow the expected format:
1. **Input**: `(B, 1, 8, 8)` tensor for position
2. **Output**: a policy distribution of shape `(B, 4672)`, and expected value of shape `(B, 1)`.

A minimal example of writing a model:
```python
from chessbot.models import BaseChessModel, ModelRegistry

@ModelRegistry.register('simple_chessnet')
class SimpleChessNet(BaseChessModel):
  """ One layer backbone and one layer prediction heads """
    
    def __init__(self):
        super().__init__()
        
        # Mini backbone
        self.backbone = nn.Linear(64, 256)

        # Policy head
        self.policy_head = nn.Linear(256, self.action_dim)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Tanh()  # Between -1 and 1 for lose, draw, win
        )

    def forward(self, x):
      """ Input is tensor of shape (B,1,8,8) """
        x             = x.view(B, -1)
        features      = self.backbone(x)            # -> (B, 256)
        action_logits = self.policy_head(features)  # -> (B, 4672)
        board_val     = self.value_head(features)   # -> (B, 1)

        return action_logits, board_val
```

The `ModelRegistry` is a helper for the library to load chess models from a path and name. The model will be registered with the name provided, or the class name if none is provided. This helps find and load models for command line tools.
## 🦾 Inference & Battling

Take your models to the battleground!  

The library depends on an **Adversarial Gym Environment** designed for two-player turn-based games, that can be used to visualize model inference. Check out the functions in [`chessbot.inference`](chessbot/inference/):

```python
from chessbot.inference import selfplay, duel

# Selfplay. Returns value in [-1,0,1] for white's outcome
model   = YourChessModel()
outcome = selfplay(model, visualize=True)

# Match between two models, use MCTS. Returns (score1,score2)
model1 = YourChessModel()
model2 = YourChessModel()
scores = duel(model1, model2, best_of=11, search=True, visualize=True)
```

Use the search flag to harness **Monte Carlo Tree Search (MCTS)** for search during inference. *MCTS training code coming soon!* The [Chess Battle GIF](#chess-battle-gif) at the beginning is an example of visualizing the game with the Chess-env, and using MCTS for test-time powered inference. 


## ✨ Getting Started

### 1. Installation: 

First install the [Adversarial Gym](https://github.com/OperationBeatMeChess/adversarial-gym) chess environment:
  ```bash
  pip install adversarial-gym
  ```
Then install ChessBot-Battleground

   ```bash
   # Either install from source...
   git clone https://github.com/KeithG33/ChessBot-Battleground.git
   cd ChessBot-Battleground
   pip install -r requirements.txt
   pip install e .  

   # Or install via pip (TODO)
   pip install ChessBot-Battleground
   ```

Once you've got the library installed check out the `chessbot` cli tool for a quick overview of things you can do:
```bash
kage@pop-os:~/chess_workspace$ chessbot --help
                                                                                                                                                                                                                                      
 Usage: chessbot [OPTIONS] COMMAND [ARGS]...                                                                                                                                                                                          
                                                                                                                                                                                                                                      
 ChessBot CLI Tool                                                                                                                                                                                                                    
                                                                                                                                                                                                                                      
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                                                                                                                                            │
│ --show-completion             Show completion for the current shell, to copy it or customize the installation.                                                                                                                     │
│ --help                        Show this message and exit.                                                                                                                                                                          │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ evaluate   Evaluate a model. Pass additional positional arguments with --model-arg and keyword arguments as a JSON string via --model-kwargs.                                                                                      │
│ download   Download a dataset from a GitHub release.                                                                                                                                                                               │
│ play       Play a game against the bot using a loaded model. Pass additional positional arguments with --model-arg and keyword arguments as a JSON string via --model-kwargs.                                                      │
│ train      Train a model using the provided configuration file and optional overrides.                                                                                                                                             │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```


### 2. Download the Dataset:
*[Releases available for download here](https://github.com/KeithG33/ChessBot-Battleground/releases)* 

The dataset is provided as a downloadable .zip file with each release. Either use the link and your browser, or the `chessbot` cli tool:
```bash
# For options and help
chessbot download --help 

# Default: download latest release to cwd if pip installed, or ChessBot-Battleground/dataset if source installed
chessbot download

# Ex: download release v0.0.0 to output_dir
chessbot download v0.0.0 \
                  --output-dir /path/to/output_dir
```

By default, the latest release will be downloaded into the `ChessBot-Battleground/dataset/` directory, or the current working directory if the package has been pip installed.  

### 3. Models & Training
After installation and downloading, it's time to write a model and let it gobble up data. Writing a model and training was covered above, so first check that out. Here I'll show the CLI version of training. First register your model, and then configure `model.path` and `model.name` to load the model. Either set this in the config file, or use the command overrides

```bash
# For options and help
chessbot train --help

# Train from config, and any overrides in command
chessbot train /path/to/config.yaml
              -o model.path path/to/model
              -o model.name YourChessModel
              -o train.epochs 10
              -o train.lr 0.001
```

Additionally, `model.args` and `model.kwargs` exist for the model init. Use a list and dictionary, respectively.


### 4. Leaderboard / Evaluation
Share your model's results on the test set. Compare your scores against the leaderboard. Once you've trained a model run the provided evaluate script to get your test set metrics.

```python
from chessbot.inference.evaluate import evaluate_model

# Load your model
model = ChessModel()

# Evaluate the model
batch_size = 3072
num_threads = 8
data_dir = 'path/to/dataset/'
evaluate_model(model, data_dir, batch_size, num_threads)
```

Or if your model is registered as "my_chessnet", using the `chessbot` cli tool:
```bash
# For options and help:
chessbot evaluate --help

chessbot evaluate "my_chessnet" \ 
                  --model-dir path/to/dir \
                  --model-weights path/to/weights.pt \
                  --data-dir path/to/dataset \
                  --batch-sz 3072 \
                  --num-threads 8 \
```

### 5. Play Your ChessBot
And once you've written a model and trained...A historically important question for humankind: *Can your model beat you?*


```bash
chessbot play "your_chessnet" \
              --model-dir /path/to/dir \
              --model-weights /path/to/weights.pt
```
<div align="center">
<img src="assets/battleground.png" style="width: 70%; height: auto;">  
  <p><em> Punishing a beautiful queen sac from a randomly initialized model ;)</em></p>
</div>

### 6. Examples

There is a simple and complete example in [examples](examples/) to get you started. Check out the `SimpleChessNet` for an example of the model interface; use `example_training.ipynb` to train the model; use `example_inference.ipynb` to either run inference with the base model, or with an MCTS wrapper for a test-time-powerup.

For actual models check out the [models](models/) directory.

Additionally, an `example_sf_datagen.ipynb` exists to show how one might add data to the dataset. Unfortunately stockfish is slow so this is a hopeful crumb that I leave for the crowd.



## 📈 Future Plans

- Clean and deduplicate the dataset
- Expand the dataset, add Stockfish generated data, get to an epic milestone of **1 billion positions**.  
- Release **MCTS training pipelines**.  
- Add enhanced tools for training and visualization and evaluation.
- Add a leaderboard


## 🛠️ Contributing

If you have ideas, improvements, or bug fixes, feel free to open an issue or submit a pull request. For any questions or further discussion, don't hesitate to reach out!
