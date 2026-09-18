"""Small standalone utilities used across the pipeline."""

import gc

import torch


def clear_memory(model=None, optimizer=None, trainer=None):
    """Delete the given objects and release GPU/CPU memory back to the OS.

    Call this between training runs (e.g. after finishing one model) to avoid
    accumulating GPU memory across the CNN / Transformer / LSTM experiments.
    """
    if model is not None:
        del model
    if optimizer is not None:
        del optimizer
    if trainer is not None:
        del trainer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU memory reserved:  {torch.cuda.memory_reserved()  / 1024**2:.2f} MB")
    else:
        print("No GPU detected — CPU memory cleared via gc.collect()")
