"""Utilities for exporting VisionXAI models for on-device inference."""


def export_models(*args, **kwargs):
    """Lazily export model assets without importing TensorFlow at package import."""
    from .export_tflite import export_models as _export_models

    return _export_models(*args, **kwargs)


__all__ = ["export_models"]
