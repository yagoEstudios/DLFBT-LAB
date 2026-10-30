import torch
from torch import nn

# All the models recieve the data in (B,561) dimensions

def adapt_input(x, model_type):
    """Receives x with shape (B, n_features) and returns it with the shape expected by each model."""
    if x.dim() != 2:
        raise ValueError(f"Expected a 2-D input (B, n_features), got shape {tuple(x.shape)}")

    match model_type:
        case "rnn" | "lstm":
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

    sequence_length only exists for compatibility with the notebook: for the DNN
    the n_features values are a flat vector, not a sequence.
    """

    def __init__(
        self,
        n_features,
        n_classes=6,
        hidden_dims=(256, 128),
        dropout=0.3,
        batch_norm=True,
        sequence_length=None,
    ):
        super().__init__()
        if sequence_length is not None and sequence_length != n_features:
            raise ValueError(
                f"DNN expects sequence_length == n_features, got {sequence_length} and {n_features}"
            )
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

        self.hidden_layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"DNN expects input of shape (batch, {self.input_dim}), got {tuple(x.shape)}"
            )
        return self.classifier(self.hidden_layers(x))


class RNNClassifier(nn.Module):
    """Vanilla recurrent network that reads the features one step at a time.

    (batch, n_features) -> (batch, n_features, 1)
    -> RNN x num_layers -> (batch, S, hidden_size)
    -> last hidden state -> (batch, hidden_size)
    -> Dropout -> Linear -> (batch, n_classes)

    The state is carried along the 561 steps, so the model assumes the order of the
    features matters, but gradients can vanish over such a long sequence.
    """

    def __init__(
        self,
        input_size,
        n_classes=6,
        hidden_size=64,
        num_layers=1,
        dropout=0.3,
    ):
        super().__init__()
        self.input_size = input_size

        # the recurrent layer reads one value per time step, so its input_size is 1
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        if x.dim() != 2 or x.size(1) != self.input_size:
            raise ValueError(
                f"RNN expects input of shape (batch, {self.input_size}), got {tuple(x.shape)}"
            )
        x = adapt_input(x, "rnn")          # (B, S, 1)
        outputs, _ = self.rnn(x)              # (B, S, hidden_size)
        x = outputs[:, -1, :]                 # (B, hidden_size)
        return self.classifier(self.dropout(x))


class CNNClassifier(nn.Module):
    """1-D convolutional network that treats the features as a sequence.

    (batch, n_features) -> (batch, 1, n_features)
    -> [Conv1d -> BatchNorm -> ReLU -> MaxPool] x len(channels)
    -> global average pooling -> (batch, channels[-1])
    -> Dropout -> Linear -> (batch, n_classes)

    The same filters slide along the sequence, so the model assumes that
    neighbouring features are related.
    """

    def __init__(
        self,
        input_size,
        n_classes=6,
        channels=(32, 64, 128),
        kernel_size=5,
        dropout=0.3,
        batch_norm=True,
    ):
        super().__init__()
        self.input_size = input_size

        layers = []
        in_channels = 1
        for out_channels in channels:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_channels, n_classes)

    def forward(self, x):
        if x.dim() != 2 or x.size(1) != self.input_size:
            raise ValueError(
                f"CNN expects input of shape (batch, {self.input_size}), got {tuple(x.shape)}"
            )
        x = adapt_input(x, "cnn")             # (B, 1, S)
        x = self.conv_layers(x)               # (B, channels[-1], S / 2**len(channels))
        x = self.pool(x).squeeze(-1)          # (B, channels[-1])
        return self.classifier(self.dropout(x))


class LSTMClassifier(nn.Module):
    """LSTM network that reads the features one step at a time.

    (batch, n_features) -> (batch, n_features, 1)
    -> LSTM x num_layers -> (batch, S, hidden_size)
    -> last hidden state -> (batch, hidden_size)
    -> Dropout -> Linear -> (batch, n_classes)

    The memory cell helps keep information over the long sequence, at the cost
    of more parameters and a slower epoch than the vanilla RNN.
    """

    def __init__(
        self,
        input_size,
        n_classes=6,
        hidden_size=64,
        num_layers=1,
        dropout=0.3,
    ):
        super().__init__()
        self.input_size = input_size

        # the recurrent layer reads one value per time step, so its input_size is 1
        self.rnn = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        if x.dim() != 2 or x.size(1) != self.input_size:
            raise ValueError(
                f"LSTM expects input of shape (batch, {self.input_size}), got {tuple(x.shape)}"
            )
        x = adapt_input(x, "lstm")          # (B, S, 1)
        outputs, _ = self.rnn(x)              # (B, S, hidden_size)
        x = outputs[:, -1, :]                 # (B, hidden_size)
        return self.classifier(self.dropout(x))