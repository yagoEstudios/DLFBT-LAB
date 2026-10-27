import torch
from torch import nn

# All the models recieve the data in (B,561) dimensions

def adapt_input(x, model_type):
    """Receives x with shape (B, n_features) and returns it with the shape expected by each model."""
    if x.dim() != 2:
        raise ValueError(f"Expected a 2-D input (B, n_features), got shape {tuple(x.shape)}")

    match model_type:
        case "rnn":
            # (B, S=n_features, input_size=1)
            return x.unsqueeze(-1)
        case "cnn":
            # (B, C=1, S=n_features)
            return x.unsqueeze(1)
        case _:
            raise ValueError(f"Unknown model_type: {model_type}")



class DNNClassifier(nn.Module):
    """Fully connected network on the flat feature vector.

    (batch, n_features)
    -> [Linear -> BatchNorm -> ReLU -> Dropout] x len(hidden_dims)
    -> Linear -> (batch, n_classes)

    Every feature gets its own independent weights, so the order of the
    features does not matter.
    """

    def __init__(
        self,
        n_features,
        n_classes=6,
        hidden_dims=(256, 128),
        dropout=0.3,
        batch_norm=True,
    ):
        super().__init__()
        self.input_dim = n_features

        layers = []
        in_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"DNN expects input of shape (batch, {self.input_dim}), got {tuple(x.shape)}"
            )
        return self.classifier(self.features(x))


class RNNClassifier:
    def __init__(self, *args, **kwargs):
        pass


class CNNClassifier:
    def __init__(self, *args, **kwargs):
        pass


class LSTMClassifier:
    def __init__(self, *args, **kwargs):
        pass