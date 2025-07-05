import glob
import importlib
import os
from chessbot.models.base import BaseChessBot
from chessbot.common import DEFAULT_MODEL_DIR


def auto_register_models():
    """
    Function to auto-import and register the models in the hall-of-fame models/ director.
    For easy access to the models.

    Args:
        model_path (str): Directory path to search for model files.
    """
    pattern = os.path.join(DEFAULT_MODEL_DIR, '**', '*.py')
    for file_path in glob.iglob(pattern, recursive=True):
        
        # Auto register the _chessbot.py files 
        if not file_path.endswith('_chessbot.py'):
            continue
                
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


class ModelRegistry:
    """
    Registry for managing and validating models derived from BaseChessModel.
    """
    _registry = {}

    @classmethod
    def register(cls, name=None):
        """
        Decorator to register a model class.
        Args:
            name (str, optional): Name to register the model with. If None, the class name is used.
        """
        def decorator(model_cls):
            register_name = name or model_cls.__name__  # Use provided name or class name if None
            if not issubclass(model_cls, BaseChessBot):
                raise ValueError(f"Model '{register_name}' must inherit from BaseChessBot.")
            if register_name in cls._registry:
                print(f"Warning: Model '{register_name}' already registered, overwriting previous registration.")
            cls._registry[register_name] = model_cls
            return model_cls

        return decorator
    
    @classmethod
    def list_models(cls):
        return list(cls._registry.keys())
  
    @classmethod
    def _load_models_from_path(cls, model_path):
        """
        Recursively loads Python files from specified directory/file path to register models using glob.
        Args:
            model_path (str): The directory from which to load Python files.
        """
        pattern = os.path.join(model_path, '**', '*.py')
        for file_path in glob.iglob(pattern, recursive=True):
            if not file_path.endswith('__init__.py'):
                filename = os.path.basename(file_path)
                spec = importlib.util.spec_from_file_location(filename, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module) 

    @classmethod
    def _load_model(cls, model_name, *init_args, **init_kwargs):
        """
        Load a model from the registry.
        Args:
            model_name (str): The name of the model to load.
            init_args (tuple): Positional arguments to pass to the model's constructor.
            init_kwargs (dict): Keyword arguments to pass to the model's constructor.

        Returns:
            An instance of the model.
        """
        try:
            ModelClass = cls._registry[model_name]
        except KeyError:
            raise KeyError(f"Model '{model_name}' is not registered. Currently available models: {cls.list_models()}")        
        return ModelClass(*init_args, **init_kwargs)

    @classmethod
    def load_model(cls, model_name=None, model_path=None, *init_args, **init_kwargs):
        """
        Load a model from the registry or a path containing models.
        Args:
            model_name (str): The name of the model to load.
            model_path (str, optional): Path containing model registration.
            init_args (tuple): Positional arguments to pass to the model's constructor.
            init_kwargs (dict): Keyword arguments to pass to the model's constructor.

        Returns:
            An instance of the model.
        """
        if model_path:
            cls._load_models_from_path(model_path)
        return cls._load_model(model_name, *init_args, **init_kwargs)

    @classmethod
    def load_with_weights(
        cls,
        model_name: str,
        weights_id: str,
        model_path: str | None = None,
        hf_filename: str = "pytorch_model.bin",
        *init_args,
        **init_kwargs,
    ):
        """Load a registered model and automatically load weights.

        Args:
            model_name: Name of the model to load.
            weights_id: Local path or HuggingFace repo id for the weights.
            model_path: Optional additional directory to search for model
                registrations.
            hf_filename: Name of the weight file on HuggingFace.
        """
        model = cls.load_model(model_name, model_path, *init_args, **init_kwargs)
        if os.path.exists(weights_id):
            model.load_weights(weights_id)
        else:
            model.load_hf_weights(weights_id, filename=hf_filename)
        return model
