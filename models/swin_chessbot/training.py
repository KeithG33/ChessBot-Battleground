import os
import chessbot
from chessbot.train import ChessHFTrainer
from chessbot.common import DEFAULT_DATASET_DIR
from chessbot.models import MODEL_REGISTRY


cwd = os.getcwd()
dataset_path = DEFAULT_DATASET_DIR

cfg = chessbot.config.get_cfg()

# Dataset
# cfg.dataset.data_path = dataset_path
# cfg.dataset.size_train = 200
# cfg.dataset.size_test = 10
# cfg.dataset.num_processes = 20
cfg.dataset.num_workers = 16
cfg.dataset.shuffle_buffer = 100_000


# Train
cfg.train.rounds = 50
cfg.train.epochs = 1 
cfg.train.batch_size = 4096
cfg.train.lr = 0.0001
cfg.train.scheduler = 'linear'
cfg.train.min_lr = 0.00005
cfg.train.warmup_lr = 0.00001
cfg.train.warmup_iters = 1000
cfg.train.compile = True
cfg.train.amp = 'bf16'
cfg.train.validation_every = 20_000

# cfg.train.output_dir = 'models/swin_chessbot/train/'
cfg.train.checkpoint_dir = 'models/swin_chessbot/train/checkpoint'
cfg.train.resume_from_checkpoint = True

cfg.logging.wandb = True
cfg.logging.wandb_run_id = 'f7pzjtef'



if __name__ == '__main__':
    model = MODEL_REGISTRY.load_model('swin_chessbot')
    trainer = ChessHFTrainer(cfg, model)
    trainer.train()