# ♟️ ChessBot Tutorial: Usage Guide

This tutorial provides an overview of the ChessBot repository, and will guide you through an end-to-end example, including any additional information that might be useful.

All examples provided are complete, allowing you to follow along and run the code yourself. Reach out if you encounter any issues!

The example model and training script are in [model/example_chessbot](../models/example_chessbot/)


## 🚀 1. Setup & Installation

**Recommended (source install):**
```bash
git clone https://github.com/KeithG33/ChessBot-Battleground.git
cd ChessBot-Battleground
pip install -r requirements.txt
pip install -e .
```

**Alternative (pip install):**
```bash
pip install git+https://github.com/KeithG33/ChessBot-Battleground.git
```

**Verification:**

```bash
>> chessbot --help

 Usage: chessbot [OPTIONS] COMMAND [ARGS]...                                                                                                                                                                                         
                                                                                                                                                                                                                                     
 ChessBot CLI Tool                                                                                                                                                                                                                   
                                                                                                                                                                                                                                     
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                                                                                                                                           │
│ --show-completion             Show completion for the current shell, to copy it or customize the installation.                                                                                                                    │
│ --help                        Show this message and exit.                                                                                                                                                                         │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ evaluate   Evaluate a model. Pass additional positional arguments with --model-arg and keyword arguments as a JSON string via --model-kwargs.                                                                                     │
│ download   Download a dataset.                                                                                                                                                                                                    │
│ play       Play a game against the bot using a loaded model. Pass additional positional arguments with --model-arg and keyword arguments as a JSON string via --model-kwargs.                                                     │
│ train      Train a model using the provided configuration file and optional overrides.                                                                                                                                            │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## ⬇️ 2.  Download the Dataset

> **Note:** Downloading the dataset is no longer required as it is now *[available to be streamed from HuggingFace.](https://ishortn.ink/chessbot-dataset)* 


For those who want to see the data, you can download from HuggingFace or use the chessbot tool:

```bash
# Download to cwd if pip installed, or ChessBot-Battleground/dataset if source installed.
chessbot download
```


## 🤖 3. Creating A ChessBot

Recall from the main [README](../README.md) that models have the format,

1. **Input**: `(B, 1, 8, 8)` tensor for position
2. **Output**: a policy distribution of shape `(B, 4672)`, and expected value of shape `(B, 1)`.

Here's a full implementation of a simple model, with the model registered as `"simple_chessbot"` and placed inside [models/example_chessbot](../models/example_chessbot/simple_chessbot.py):

```python
@ModelRegistry.register("simple_chessbot")
class SimpleChessBot(BaseChessBot):  
    def __init__(self):
        super().__init__()    

        # Simple mlp backbone
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 256),  # Flatten the 8x8 board and process
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(64, self.action_dim)
        ) 

        # Value head
        self.value_head  = nn.Sequential(
            nn.Linear(64, 1),
            nn.Tanh()  # Output a value between -1 and 1
        )

    def forward(self, x):
        features      = self.layers(x)
        action_logits = self.policy_head(features)
        board_val     = self.value_head(features)
        return action_logits, board_val
```

Now we can automatically load this model for training or inference by setting `cfg.model.name = "simple_chessbot"`, and with `cfg.model.args` or `cfg.model.kwargs` as needed.

>**Note:** models in the `models/` directory will be auto-registered for easy loading


## 🏋️ 4. Training Your ChessBot

#### Python

Training in python means you need to create a config by loading it or creating one at runtime. Or a combination of these solutions. Here we will show everything. The below code is [train.py](../models/example_chessbot/train.py)

```python
from omegaconf import OmegaConf
from chessbot.train import config
from chessbot.train import ChessTrainer

# Get default OmegaConf cfg
cfg = config.get_cfg()

# Alternatively, load some config overrides (or an entire config) from a file
cfg_load = OmegaConf.load('models/example_chessbot/config.yaml')

# Override cfg with cfg_load, and add any new keys
cfg = OmegaConf.merge(cfg, cfg_load)

cfg.train.epochs = 25 # num epochs on sampled dataset
cfg.train.batch_size = 128
cfg.train.lr = 0.001
cfg.train.output_dir = 'models/example_chessbot/output/'
cfg.dataset.num_workers = 8
cfg.dataset.shuffle_buffer = 100_000

# Option 1: Load model from registry
cfg.model.name = "simple_chessbot"
cfg.model.kwargs = {"hidden_dim": 512}
trainer = ChessTrainer(cfg, load_model_from_config=True)
trainer.train()

# Option 2: Load model from path
from simple_chessbot import SimpleChessBot
model = SimpleChessBot(hidden_dim=512)
trainer = ChessTrainer(cfg, model)
trainer.train()
```

The training output directory will store the config used and the best model, latest model, and the complete training state for resuming from checkpoint in a `checkpoint/` directory. 

To resume from checkpoint set `cfg.train.checkpoint_dir` to this `checkpoint/` directory. And to reuse the same training output directory set `resume_from_checkpoint=True`.
```python
# Use training state from previous train checkpoint (weights, optimizer, scheduler) 
cfg.train.checkpoint_dir = 'previous/train_dir/checkpoint/'

# Reuse previous train directory
cfg.train.resume_from_checkpoint = True

# Train ...
```

#### Using The CLI
We can also use the command-line or a bash script to start training. See [train.sh](../models/example_chessbot/train.sh) for an example:


```bash
# Train from config, and any overrides in command
chessbot train models/example_chessbot/config.yaml \
              -o model.name=simple_chessbot \
              -o train.epochs=10 \
              -o train.lr=0.001 \
              -o train.batch_size=64 \
              -o train.shuffle_buffer=100_000 \
```

Loading from config with CLI functions the same as with python. Use `cfg.model.name`, and `cfg.model.path` if it is not inside the `models/` directory. 


#### Configs
Definitely check out the [config](../chessbot/train/config.yaml) for all the options, but I'll highlight some of the more valuable ones here.
1. `cfg.train.grad_accum` - number of gradient accumulation steps to perform before updating
2. `cfg.train.grad_clip` - default is norm value of 1.0
3. `cfg.train.compile` - torch.compile your model (currently using INDUCTOR backend). True or False
4. `cfg.train.amp` - defaults to 'no', but can be set to: 'fp16', 'bf16', 'fp8', 'fp32'
5. `cfg.train.optimizer` - Use `adamw` or `sgd` or any optimizer from the `timm` package
5. `cfg.train.warmup_*` - configure warmup learning rate for training
6. `cfg.train.validation_every` - configure how often (in iters) to validate during training
7. `cfg.logging.wandb_*` - configure wandb logging. Also see `cfg.logging.log_every` for logging frequency


## 📊 5. Evaluating Your ChessBot

One warning is that the test set is quite large and evaluation may take some time depending on model and hardware.
```python
from chessbot.inference import evaluate_model
from chessbot.common import DEFAULT_DATASET_DIR
from simple_chessbot import SimpleChessBot

if __name__ == "__main__":    
    dataset_dir = DEFAULT_DATASET_DIR
    model = SimpleChessBot(hidden_dim=512)
    # Need to train to get weights
    # model.load_weights('models/example_chessbot/output/model_best/pytorch_model.bin')
    evaluate_model(
        model,
        dataset_dir=DEFAULT_DATASET_DIR,
        device="cuda",
        batch_size=64, # batch size
    )
```


#### Using The CLI
```bash
chessbot evaluate "simple_chessbot" \
                  --model-weights models/example_chessbot/output/model_best/pytorch_model.bin \
                  --batch-sz 3072 \
                  --num-threads 8 \
                  --num_chunks 0
```

Don't forget to share your scores! The evaluation script covers:
1. Policy Metrics (classification - accuracy, top5, top10, cross entropy)
2. Value Metrics (regression - mse, mae)

***Note***: if torch.compile is used for training the output weights may have mismatching keys. Use `model.load_weights('weights.bin')` to automatically handle this.

## 🥊 6. Battling with ChessBots
The `chessbot.inference` package contains functions for playing games with your trained bots.

### 🤖 Self-play

```python
from chessbot.inference import selfplay
from simple_chessbot import SimpleChessBot

model = SimpleChessBot(hidden_dim=512)
# model.load_state_dict(torch.load('pytorch_model.bin'))

outcome = selfplay(
  model, 
  search=True, # Use MCTS search
  num_sims=250, # How many simulations to run
  visualize=True # Display the game
)
```

The outcome will correspond to losing-drawing-winning with white.

### ⚔️ Play a match between two models
```python
from chessbot.inference import run_match
from simple_chessbot import SimpleChessBot

p1 = SimpleChessBot(hidden_dim=512)
# p1.load_state_dict(torch.load('pytorch_model1.bin'))

p2 = SimpleChessBot(hidden_dim=512)
# p2.load_state_dict(torch.load('pytorch_model2.bin'))

scores = run_match(
  p1, # player1 model
  p2, # player2 model
  best_of=7, # Best-of 
  search=False, # Use MCTS search
  num_sims=250, # Num sims if searching
  visualize=True # Display the game
  sample=False, # Sample or select best from policy distribution
)
```

The returned `scores` will be a tuple of `(p1_score, p2_score)` with the usual +1 for win, 0.5 for draw, and 0 for loss. 

## 🎮 7. GamePlay App

Of course the most important thing is playing your bot and seeing it it beats you.

```bash
chessbot play "simple_chessbot" \
              --model-weights /path/to/weights.bin \
              --model-kwargs '{"hidden_dim": 512}' \ # optional args and kwargs
```

This will start a simple game app to play against your model. You can also play against the
other hall-of-fame models in the [models/](../models/) directory using their HuggingFace weights:

```bash
chessbot play "sgu_chessbot" \
              --model-weights KeithG33/sgu_chessbot \
```

You can also run automated games from the command line:

```bash
# Selfplay
chessbot selfplay "sgu_chessbot" --model-weights KeithG33/sgu_chessbot --search

# Best-of match
chessbot play-match "sgu_chessbot" "simple_chessbot" \
               --player1-weights KeithG33/sgu_chessbot \
               --player2-weights /path/to/simple_weights.pt
```
## 💾 8. Generating Data

Future goals are to add lots more data using stockfish as the teacher. However generating data with my personal computer is prohibitively slow so I am leaving code as a little breadcrumb/bait for the community.

See [example_sf_datagen.ipynb](../examples/example_sf_datagen.ipynb)

--- 

\
🎉 **Happy training and battling!**

If you encounter issues, feel free to reach out!

