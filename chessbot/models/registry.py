import importlib
import os
from chessbot.models.base import BaseChessModel


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
            if not issubclass(model_cls, BaseChessModel):
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
            raise KeyError(f"Model '{name}' is not registered.")
        return cls._registry[name]

    @classmethod
    def list_models(cls):
        return list(cls._registry.keys())
  
    @classmethod
    def load_models_from_directory(cls, directory_path):
        """
        Automatically loads all Python files in the specified directory to register models.
        Args:
            directory_path (str): The directory from which to load Python files.
        """
        for filename in os.listdir(directory_path):
            if filename.endswith('.py') and not filename.startswith('__'):
                file_path = os.path.join(directory_path, filename)
                spec = importlib.util.spec_from_file_location(filename, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

    @classmethod
    def load_model(cls, model_name, *init_args, **init_kwargs):
        """
        Load a model from the registry.
        Args:
            model_name (str): The name of the model to load.
            init_args (tuple): Positional arguments to pass to the model's constructor.
            init_kwargs (dict): Keyword arguments to pass to the model's constructor.

        Returns:
            An instance of the model.
        """
        if model_name not in cls._registry:
            raise KeyError(f"Model '{model_name}' is not registered.")
        ModelClass = cls._registry[model_name]
        model_instance = ModelClass(*init_args, **init_kwargs)
        return model_instance