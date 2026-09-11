from abc import ABC, abstractmethod
import tensorflow as tf

class BaseEncoder(tf.keras.Model, ABC):
    """Abstract base class for captioning encoder models."""
    pass

class BaseDecoder(tf.keras.Model, ABC):
    """Abstract base class for captioning decoder models."""
    @abstractmethod
    def reset_state(self, batch_size: int) -> tf.Tensor:
        """Resets the decoder recurrent hidden states."""
        pass
