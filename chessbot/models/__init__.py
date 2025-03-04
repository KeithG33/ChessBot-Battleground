from .base import BaseChessBot
from .registry import ModelRegistry


def align_state_dict(state_dict, prefix="_orig_mod.", new_prefix=""):
    """ Fix keys of a compiled model's state_dict, which have _orig_mod. prefixed to start """
    new_state_dict = {}

    # replace the specified prefix to each key
    for key, value in state_dict.items():
        new_key = key.replace(prefix, new_prefix)
        new_state_dict[new_key] = value
    return new_state_dict