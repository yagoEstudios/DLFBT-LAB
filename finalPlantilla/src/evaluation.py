"""Evaluation utilities: metrics, parameter counts, timing and model size."""

import io
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


def _model_device(model):
    return next(model.parameters()).device


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def predict(model, X, batch_size=512, device=None, return_probabilities=False):
    """Predicted labels (and optionally softmax probabilities) for `X`."""
    device = torch.device(device) if device is not None else _model_device(model)
    model = model.to(device).eval()

    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    logits = []
    for start in range(0, len(X), batch_size):
        batch = X[start : start + batch_size].to(device)
        logits.append(model(batch).cpu())
    logits = torch.cat(logits)

    predictions = logits.argmax(dim=1).numpy()
    if return_probabilities:
        return predictions, torch.softmax(logits, dim=1).numpy()
    return predictions


def per_class_metrics(y_true, y_pred, class_names=None):
    """Per-class precision, recall, F1 and support as a DataFrame."""
    labels = np.arange(len(class_names)) if class_names is not None else None
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    index = class_names if class_names is not None else np.unique(y_true)
    return pd.DataFrame(
        {"precision": precision, "recall": recall, "f1": f1, "support": support},
        index=pd.Index(index, name="class"),
    )


def evaluate_model(model, X, y, batch_size=512, device=None, class_names=None):
    """Compute the global and per-class classification metrics.

    Macro metrics average the per-class scores with equal weight; weighted
    metrics weight each class by its support (number of true samples).
    """
    y = np.asarray(y)
    predictions, probabilities = predict(
        model, X, batch_size=batch_size, device=device, return_probabilities=True
    )

    return {
        "accuracy": accuracy_score(y, predictions),
        "macro_precision": precision_score(y, predictions, average="macro", zero_division=0),
        "macro_recall": recall_score(y, predictions, average="macro", zero_division=0),
        "macro_f1": f1_score(y, predictions, average="macro", zero_division=0),
        "weighted_precision": precision_score(y, predictions, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y, predictions, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y, predictions, average="weighted", zero_division=0),
        "per_class": per_class_metrics(y, predictions, class_names),
        "confusion_matrix": confusion_matrix(y, predictions),
        "predictions": predictions,
        "probabilities": probabilities,
    }


def count_parameters(model, trainable_only=True):
    """Number of (trainable) parameters of `model`."""
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad or not trainable_only
    )


@torch.no_grad()
def measure_inference_stats(model, X, batch_size=512, device=None, n_repeats=10, warmup=2):
    """Time a full forward pass over `X`.

    The model is run `warmup` times first (CUDA context, cuDNN kernel
    selection, caches), then `n_repeats` timed passes over the whole set.
    Data is moved to the device beforehand so only computation is timed.
    """
    device = torch.device(device) if device is not None else _model_device(model)
    model = model.to(device).eval()

    X = torch.as_tensor(np.asarray(X), dtype=torch.float32).to(device)
    batches = [X[i : i + batch_size] for i in range(0, len(X), batch_size)]

    def run():
        for batch in batches:
            model(batch)

    for _ in range(warmup):
        run()

    times = []
    for _ in range(n_repeats):
        _sync(device)
        start = time.perf_counter()
        run()
        _sync(device)
        times.append(time.perf_counter() - start)

    times = np.array(times)
    return {
        "total_time_s": times.mean(),
        "total_time_std_s": times.std(),
        "time_per_sample_ms": 1000 * times.mean() / len(X),
        "samples_per_second": len(X) / times.mean(),
    }


def measure_inference_time(model, X, **kwargs):
    """Mean wall-clock time (seconds) to predict the whole of `X`."""
    return measure_inference_stats(model, X, **kwargs)["total_time_s"]


def model_size_mb(model):
    """Size of the serialised ``state_dict`` in megabytes."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / 1024**2
