from typing import Dict, Type, List, Optional
from .base import BaseEncoder, BaseDecoder

class ModelRegistry:
    """
    Registry for image captioning encoders and decoders.
    Allows registering custom models via decorators and creating them by name.
    """
    _encoders: Dict[str, Type[BaseEncoder]] = {}
    _decoders: Dict[str, Type[BaseDecoder]] = {}

    @classmethod
    def register_encoder(cls, name: str):
        """Decorator to register a custom encoder class."""
        def decorator(subclass: Type[BaseEncoder]):
            cls._encoders[name] = subclass
            subclass.model_name = name
            return subclass
        return decorator

    @classmethod
    def register_decoder(cls, name: str):
        """Decorator to register a custom decoder class."""
        def decorator(subclass: Type[BaseDecoder]):
            cls._decoders[name] = subclass
            subclass.model_name = name
            return subclass
        return decorator

    @classmethod
    def create_encoder(cls, name: str, *args, **kwargs) -> BaseEncoder:
        """Instantiate a registered encoder model by name."""
        if name not in cls._encoders:
            raise ValueError(f"Encoder '{name}' is not registered. Available: {list(cls._encoders.keys())}")
        return cls._encoders[name](*args, **kwargs)

    @classmethod
    def create_decoder(cls, name: str, *args, **kwargs) -> BaseDecoder:
        """Instantiate a registered decoder model by name."""
        if name not in cls._decoders:
            raise ValueError(f"Decoder '{name}' is not registered. Available: {list(cls._decoders.keys())}")
        return cls._decoders[name](*args, **kwargs)

    @classmethod
    def get_registered_names(cls) -> List[str]:
        """Get names of all models that have both registered encoders and decoders."""
        return list(set(cls._encoders.keys()) & set(cls._decoders.keys()))
