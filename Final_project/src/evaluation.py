import io
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import(
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score
)

def get_model_device(model):
    """Return the device where the model's parameters are stored."""
    return next(model.parameters()).device

def sync(device):
    "Wait for the pending GPU operations to finish"
    if device.type == "cuda":
        torch.cude.synchronize(device)

@torch.no_grad()
def predict(model,X,batch_size=512,device=None, return_probabilities=False):
    """"Predict class labels for X in batches (optionally also softmax probabilities)."""
    if device is None:
        device = get_model_device(model)
    device = torch.device(device)

    X = torch.as_tensor(X,dtype=torch.float32)
    logits = []
    
    for start in range(0,X.shape[0],batch_size):
        batch = X[start:start + batch_size].to(device)
        pred = model(batch)
        pred = pred.cpu()
        logits.append(pred)
    logits = torch.cat(logits)
    predictions = logits.argmax(dim=1).numpy()
    
    if return_probabilities:
        return predictions,torch.softmax(logits,dim=1).numpy()
    
    return predictions

def per_class_metrics(y_true, y_pred, class_names=None):
    """Return a DataFrame with precision, recall, F1 and support for each class."""
    if class_names is not None:
        labels = np.arange(len(class_names))
        index = class_names
    else:
        labels = None
        index = np.unique(y_true)
        
    precision,recall,f1,support = precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        zero_division=0)
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support
    }
    
    row_index = pd.Index(index,name="class")
    table = pd.DataFrame(metrics,index=row_index)
    
    return table
    


def evaluate_model(model,X,y,batch_size=512,device=None,class_names=None):
    """Evaluate the model on (X, y) and return global and per-class classification metrics.
    Macro metrics (class scores with equal weight)
    Weighted metrics (weight each class by its support)
    
    returns a dictionary with the metrics
    """
    y = np.asarray(y)
    
    predictions,probabilities = predict(model,X,batch_size=batch_size,device=device,return_probabilities=True)
    
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


def count_parameters(model):
    """Return the total number of parameters of the model."""
    total = 0
    for p in model.parameters():          # cada p es un tensor de pesos (una matriz W o un vector b)
        total += p.numel()            
    return total

@torch.no_grad()
def measure_inference_time(model, X, batch_size=512, device=None, n_repeats=10, warmup=2):
    """Return the mean time in seconds to predict the whole of X."""
    if device is None:
        device = get_model_device(model)
    device = torch.device(device)

    model = model.to(device)
    model.eval()

    
    X = torch.as_tensor(X, dtype=torch.float32)
    X = X.to(device)

    # Warmup (CUDA initialisation, caches)
    for _ in range(warmup):
        for start in range(0, X.shape[0], batch_size):
            model(X[start : start + batch_size])

    times = []
    for _ in range(n_repeats):
        sync(device)
        start_time = time.perf_counter()

        for start in range(0, X.shape[0], batch_size):
            model(X[start : start + batch_size])

        sync(device)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    return np.mean(times)
