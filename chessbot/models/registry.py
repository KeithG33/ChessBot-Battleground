import glob
import importlib
import os
from chessbot.models.base import BaseChessBot
from chessbot.common import DEFAULT_MODEL_DIR

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
                raise ValueError(f"Model '{register_name}' must inherit from BaseChessModel.")
            if register_name in cls._registry:
                print(f"Warning: Model '{register_name}' already registered, overwriting previous registration.")
            cls._registry[register_name] = model_cls
            return model_cls

        return decorator
    
    @classmethod
    def exists(cls, name):
        return name in cls._registry
        
    @classmethod
    def get(cls, name):
        if name not in cls._registry:
            raise KeyError(f"Model '{name}' is not registered. Currently available models: {cls.list_models()}")
        return cls._registry[name]

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
        ModelClass = cls.get(model_name)
        model_instance = ModelClass(*init_args, **init_kwargs)
        return model_instance
    
    @classmethod
    def load_model(cls, model_name, model_path=None, *init_args, **init_kwargs):
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
        if model_name and not model_path and not cls.exists(model_name):
            model_path = DEFAULT_MODEL_DIR

        if model_path:
            cls._load_models_from_path(model_path)
        return cls._load_model(model_name, *init_args, **init_kwargs)
