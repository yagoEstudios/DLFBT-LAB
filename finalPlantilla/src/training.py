"""Generic training loop shared by all architectures."""

import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .utils import get_device, set_seed


def _to_tensor(array, dtype):
    if isinstance(array, torch.Tensor):
        return array.to(dtype)
    return torch.as_tensor(np.asarray(array), dtype=dtype)


def make_loader(X, y, batch_size=64, shuffle=False, seed=None):
    """Wrap NumPy arrays/tensors in a `DataLoader`."""
    dataset = TensorDataset(_to_tensor(X, torch.float32), _to_tensor(y, torch.long))
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=None):
    """One pass over `loader` with parameter updates. Returns (loss, accuracy)."""
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * y_batch.size(0)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        n += y_batch.size(0)

    return total_loss / n, correct / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Loss and accuracy on `loader` without updating the model."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        logits = model(X_batch)
        total_loss += criterion(logits, y_batch).item() * y_batch.size(0)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        n += y_batch.size(0)

    return total_loss / n, correct / n


class EarlyStopping:
    """Stop when the monitored validation loss has not improved for `patience` epochs.

    The weights of the best epoch are kept in memory so they can be restored.
    """

    def __init__(self, patience=8, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_state = None
        self.counter = 0

    def step(self, val_loss, model, epoch):
        """Update the state; returns True when training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
        return self.patience is not None and self.counter >= self.patience


def fit(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=50,
    batch_size=64,
    learning_rate=1e-3,
    weight_decay=0.0,
    patience=8,
    min_delta=0.0,
    seed=None,
    device=None,
    grad_clip=1.0,
    restore_best_weights=True,
    checkpoint_path=None,
    verbose=True,
):
    """Train `model` with Adam + cross-entropy and early stopping on validation loss.

    Parameters
    ----------
    model : nn.Module returning logits of shape (batch, n_classes).
    X_train, y_train, X_val, y_val : arrays or tensors.
    epochs : maximum number of epochs.
    batch_size, learning_rate, weight_decay : optimisation settings.
    patience : epochs without improvement of the validation loss before stopping
        (``None`` disables early stopping).
    min_delta : minimum decrease of the validation loss counted as improvement.
    seed : if given, seeds every RNG and the shuffling order of the loader.
    device : "cuda", "cpu" or None (auto).
    grad_clip : max global gradient norm (``None`` to disable). Clipping matters
        for recurrent models, whose gradients can explode over long sequences.
    restore_best_weights : load the weights of the best validation epoch at the end.
    checkpoint_path : optional path where the best ``state_dict`` is saved.

    Returns
    -------
    dict with keys ``model``, ``history``, ``training_time``,
    ``epochs_completed``, ``best_epoch``, ``best_val_loss``.
    """
    if seed is not None:
        set_seed(seed)

    device = get_device(device)
    model = model.to(device)

    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True, seed=seed)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    stopper = EarlyStopping(patience=patience, min_delta=min_delta)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "epoch_time": [],
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    _sync(device)
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        _sync(device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        history["epoch_time"].append(time.perf_counter() - epoch_start)

        stop = stopper.step(val_loss, model, epoch)
        if stopper.best_epoch == epoch and checkpoint_path is not None:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(stopper.best_state, checkpoint_path)

        if verbose:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
                f"{history['epoch_time'][-1]:.2f}s"
                + (" *" if stopper.best_epoch == epoch else "")
            )

        if stop:
            if verbose:
                print(
                    f"Early stopping at epoch {epoch}; "
                    f"best epoch {stopper.best_epoch} (val loss {stopper.best_loss:.4f})"
                )
            break

    _sync(device)
    training_time = time.perf_counter() - start

    if restore_best_weights and stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2

    return {
        "model": model,
        "history": history,
        "training_time": training_time,
        "epochs_completed": len(history["train_loss"]),
        "best_epoch": stopper.best_epoch,
        "best_val_loss": stopper.best_loss,
        "peak_gpu_memory_mb": peak_memory_mb,
    }
