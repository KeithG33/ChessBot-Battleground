from .base import BaseChessBot, align_state_dict
from .registry import ModelRegistry, auto_register_models

from chessbot.common import DEFAULT_MODEL_DIR

auto_register_models()
MODEL_REGISTRY = ModelRegistry

