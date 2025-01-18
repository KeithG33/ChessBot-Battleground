from chessbot.models.base import BaseModel


class ModelRegistry:
    """
    Registry for managing and validating models derived from BaseModel.
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
            register_name = name or model_cls.__name__
            if not issubclass(model_cls, BaseModel):
                raise ValueError(f"Model '{register_name}' must inherit from BaseModel.")
            if register_name in cls._registry:
                raise ValueError(f"Model '{register_name}' is already registered.")
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
