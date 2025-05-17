import os
import chessbot
from chessbot.train import HFChessTrainer
from chessbot.common import DEFAULT_DATASET_DIR
from chessbot.models import MODEL_REGISTRY


cwd = os.getcwd()
dataset_path = DEFAULT_DATASET_DIR

cfg = chessbot.config.get_cfg()


# Train
cfg.train.epochs = 50 
cfg.train.batch_size = 4096
cfg.train.lr = 0.0001
cfg.train.scheduler = 'linear'
cfg.train.min_lr = 0.00005
cfg.train.warmup_lr = 0.00001
cfg.train.warmup_iters = 1000
cfg.train.compile = True
cfg.train.amp = 'bf16'
cfg.train.validation_every = 15_000

cfg.train.checkpoint_dir = 'models/swin_chessbot/train/checkpoint'
cfg.train.resume_from_checkpoint = True

cfg.dataset.num_workers = 20
cfg.dataset.num_test_samples = 17_500_000
cfg.dataset.shuffle_buffer = 5_000_000

cfg.logging.wandb = True
cfg.logging.wandb_run_id = 'f7pzjtef'



if __name__ == '__main__':
    model = MODEL_REGISTRY.load_model('swin_chessbot')
    trainer = HFChessTrainer(cfg, model)
    trainer.train()