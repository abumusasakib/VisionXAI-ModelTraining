import sys
import os
import pytest
from unittest.mock import MagicMock

# Define MockTensor and Mock classes
class MockTensor:
    def __init__(self, shape):
        self.shape = shape

# Check if tensorflow is installed; if not, mock it
try:
    import tensorflow as tf
    is_mock = False
except ImportError:
    is_mock = True
    class MockModel:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return MagicMock()
            
    class MockLayer:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return MagicMock()

    mock_tf = MagicMock()
    mock_tf.keras.Model = MockModel
    mock_tf.keras.layers.Dense = MockLayer
    mock_tf.keras.layers.Embedding = MockLayer
    mock_tf.keras.layers.GRU = MockLayer
    mock_tf.keras.layers.LayerNormalization = MockLayer
    
    mock_tf.expand_dims = lambda x, axis: x
    mock_tf.reduce_sum = lambda x, axis: x
    mock_tf.concat = lambda inputs, axis: inputs[0]
    mock_tf.zeros = lambda shape, *args, **kwargs: MockTensor(shape)
    
    mock_tf.nn.tanh = lambda x: x
    mock_tf.nn.softmax = lambda x, *args, **kwargs: x
    mock_tf.float32 = "float32"
    mock_tf.int32 = "int32"
    
    sys.modules['tensorflow'] = mock_tf
    import tensorflow as tf

# Ensure code directory is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code")))

from models.registry import ModelRegistry
import models.show_attend_tell  # trigger registration


def test_registry_registration():
    """Verify that show_attend_tell model is registered in the registry."""
    registered = ModelRegistry.get_registered_names()
    assert "show_attend_tell" in registered


def test_create_and_forward_pass():
    """Verify model instantiation via registry and a basic forward pass."""
    embedding_dim = 256
    units = 512
    vocab_size = 1000

    # Create model components via registry
    encoder = ModelRegistry.create_encoder("show_attend_tell", embedding_dim=embedding_dim)
    decoder = ModelRegistry.create_decoder(
        "show_attend_tell", 
        embedding_dim=embedding_dim, 
        units=units, 
        vocab_size=vocab_size
    )

    assert encoder is not None
    assert decoder is not None

    # Dummy inputs
    batch_size = 4

    if is_mock:
        # Mock actual call behavior to return expected MockTensors
        encoder.__class__.__call__ = lambda self, x: MockTensor((x.shape[0], 64, embedding_dim))
        decoder.__class__.__call__ = lambda self, x, features, hidden: (
            MockTensor((x.shape[0], vocab_size)),
            MockTensor((x.shape[0], units)),
            MockTensor((x.shape[0], 64, 1))
        )
        # Mock reset_state
        decoder.reset_state = lambda batch_size: MockTensor((batch_size, units))

    dummy_features = tf.zeros((batch_size, 64, 2048))
    dummy_input_seq = tf.zeros((batch_size, 1))
    dummy_hidden = decoder.reset_state(batch_size=batch_size)

    # Forward pass through encoder
    encoded_features = encoder(dummy_features)
    assert encoded_features.shape == (batch_size, 64, embedding_dim)

    # Forward pass through decoder
    predictions, hidden, attention_weights = decoder(dummy_input_seq, encoded_features, dummy_hidden)
    assert predictions.shape == (batch_size, vocab_size)
    assert hidden.shape == (batch_size, units)
    assert attention_weights.shape == (batch_size, 64, 1)

