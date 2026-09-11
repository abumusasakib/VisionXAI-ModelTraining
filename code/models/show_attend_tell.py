import tensorflow as tf
from .base import BaseEncoder, BaseDecoder
from .registry import ModelRegistry

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units, kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.W2 = tf.keras.layers.Dense(units, kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.V = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(1e-4))

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        attention_hidden_layer = tf.nn.tanh(
            self.W1(features) + self.W2(hidden_with_time_axis)
        )
        score = self.V(attention_hidden_layer)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# CNN Encoder — Formal Architecture

# Implements the formal spec from `model_architecture.tex`:

# $$\mathbf{F} = \text{LayerNorm}(\text{ReLU}(W_{\text{enc}} \cdot \mathbf{X} + b_{\text{enc}}))$$

# - Input:  $\mathbf{X} \in \mathbb{R}^{B \times K \times 2048}$ — InceptionV3 spatial features ($K=64$ ROIs)
# - Output: $\mathbf{F} \in \mathbb{R}^{B \times K \times d'}$ — projected ROI embeddings ($d'=256$)

# **LayerNorm toggle**: Set `use_layer_norm = True` (formal spec default). If LayerNorm hurts convergence, switch to `False` and retrain to compare validation loss.

@ModelRegistry.register_encoder("show_attend_tell")
class CNN_Encoder(BaseEncoder):
    def __init__(self, embedding_dim, use_layer_norm=True):
        super(CNN_Encoder, self).__init__()
        self.use_layer_norm = use_layer_norm
        self.fc = tf.keras.layers.Dense(
            embedding_dim,
            name="roi_projection",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )
        if use_layer_norm:
            self.layer_norm = tf.keras.layers.LayerNormalization(
                axis=-1, name="roi_layer_norm"
            )
        self.dropout = tf.keras.layers.Dropout(0.3)
        print(f"🔧 CNN_Encoder: LayerNorm={'ON ✅' if use_layer_norm else 'OFF ❌'}")

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        if self.use_layer_norm:
            x = self.layer_norm(x)
        x = self.dropout(x)
        return x


@ModelRegistry.register_decoder("show_attend_tell")
class RNN_Decoder(BaseDecoder):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(1e-4)
        )
        self.gru = tf.keras.layers.GRU(
            self.units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            recurrent_regularizer=tf.keras.regularizers.l2(1e-4)
        )
        self.fc1 = tf.keras.layers.Dense(self.units, kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.fc2 = tf.keras.layers.Dense(vocab_size, dtype="float32", kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.attention = BahdanauAttention(self.units)
        self.dropout = tf.keras.layers.Dropout(0.3)

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.dropout(x)
        x = self.fc2(x)
        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
