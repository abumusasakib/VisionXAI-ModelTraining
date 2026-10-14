"""Export the trained captioning model into Flutter-compatible TFLite assets.

The exporter produces two models: an InceptionV3 feature extractor and a single
decoder-step model. Keeping the decoder state explicit lets a Flutter client run
greedy decoding while retaining the attention map for each generated token.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict

import tensorflow as tf

from models import CNN_Encoder, RNN_Decoder


class FeatureExtractor(tf.Module):
    def __init__(self, model: tf.keras.Model):
        super().__init__()
        self.model = model

    @tf.function(input_signature=[
        tf.TensorSpec([1, 299, 299, 3], tf.float32, name="input_image")
    ])
    def __call__(self, input_image: tf.Tensor) -> tf.Tensor:
        return self.model(input_image, training=False)


class DecoderStep(tf.Module):
    def __init__(self, encoder: CNN_Encoder, decoder: RNN_Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @tf.function(input_signature=[
        tf.TensorSpec([1, 1], tf.int32, name="dec_input"),
        tf.TensorSpec([1, 64, 2048], tf.float32, name="image_features"),
        tf.TensorSpec([1, 512], tf.float32, name="hidden"),
    ])
    def __call__(
        self,
        dec_input: tf.Tensor,
        image_features: tf.Tensor,
        hidden: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        encoded_features = self.encoder(image_features, training=False)
        predictions, next_hidden, attention_weights = self.decoder(
            dec_input, encoded_features, hidden, training=False
        )
        return {
            "predictions": predictions,
            "next_hidden": next_hidden,
            "attention_weights": attention_weights,
        }


def _load_tokenizer(tokenizer_path: Path) -> Any:
    with tokenizer_path.open("rb") as handle:
        return pickle.load(handle)


def _write_vocabulary(tokenizer: Any, output_path: Path) -> None:
    vocabulary = {str(word): int(index) for word, index in tokenizer.word_index.items()}
    vocabulary.setdefault("<pad>", 0)
    output_path.write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _restore_models(checkpoint: Path, tokenizer: Any) -> tuple[CNN_Encoder, RNN_Decoder]:
    vocab_size = max(tokenizer.word_index.values(), default=0) + 1
    encoder = CNN_Encoder(256)
    decoder = RNN_Decoder(256, 512, vocab_size)
    encoder(tf.zeros([1, 64, 2048], tf.float32), training=False)
    decoder(
        tf.zeros([1, 1], tf.int32),
        tf.zeros([1, 64, 256], tf.float32),
        tf.zeros([1, 512], tf.float32),
        training=False,
    )
    optimizer = tf.keras.optimizers.Adam()
    status = tf.train.Checkpoint(
        encoder=encoder, decoder=decoder, optimizer=optimizer
    ).restore(str(checkpoint))
    status.expect_partial()
    return encoder, decoder


def _convert_feature_extractor(output_path: Path) -> None:
    backbone = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet")
    wrapper = FeatureExtractor(tf.keras.Model(backbone.input, backbone.layers[-1].output))
    concrete = wrapper.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], wrapper)
    output_path.write_bytes(converter.convert())


def _convert_decoder(
    encoder: CNN_Encoder, decoder: RNN_Decoder, output_path: Path
) -> None:
    wrapper = DecoderStep(encoder, decoder)
    concrete = wrapper.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], wrapper)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    output_path.write_bytes(converter.convert())


def _verify_model(path: Path) -> list[dict[str, Any]]:
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return [
        {"name": detail["name"], "shape": detail["shape"].tolist()}
        for detail in interpreter.get_input_details()
    ]


def export_models(
    checkpoint: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    include_feature_extractor: bool = True,
) -> dict[str, Path]:
    """Export model assets and return their paths.

    ``checkpoint`` may be a TensorFlow checkpoint prefix such as
    ``results/best_model/ckpt-3``. The output directory is suitable for copying
    directly into a Flutter app's asset directory.
    """
    checkpoint = Path(checkpoint)
    tokenizer_path = Path(tokenizer_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(tokenizer_path)
    _write_vocabulary(tokenizer, output_dir / "vocab.json")
    encoder, decoder = _restore_models(checkpoint, tokenizer)

    outputs: dict[str, Path] = {"vocab": output_dir / "vocab.json"}
    if include_feature_extractor:
        outputs["feature_extractor"] = output_dir / "feature_extractor.tflite"
        _convert_feature_extractor(outputs["feature_extractor"])
    outputs["decoder"] = output_dir / "decoder.tflite"
    _convert_decoder(encoder, decoder, outputs["decoder"])

    for name, path in outputs.items():
        if path.suffix == ".tflite":
            print(f"{name}: {path} inputs={_verify_model(path)}")
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="TensorFlow checkpoint prefix")
    parser.add_argument("--tokenizer", required=True, help="Pickled Keras tokenizer")
    parser.add_argument("--output-dir", required=True, help="Flutter asset output directory")
    parser.add_argument(
        "--skip-feature-extractor",
        action="store_true",
        help="Export only decoder.tflite and vocab.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    export_models(
        checkpoint=args.checkpoint,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        include_feature_extractor=not args.skip_feature_extractor,
    )
