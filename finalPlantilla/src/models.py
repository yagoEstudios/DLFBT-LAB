"""Model definitions for the UCI-HAR architecture comparison.

Every classifier accepts either

* a flat batch ``(batch, sequence_length)`` -- one channel per time step, or
* a sequential batch ``(batch, sequence_length, n_features)``

and returns unnormalised class scores (logits) of shape ``(batch, n_classes)``.
Softmax is applied inside ``nn.CrossEntropyLoss`` during training.
"""

import torch
from torch import nn


def _as_sequence(x):
    """Return ``x`` as ``(batch, time, features)``.

    A 2-D input ``(batch, time)`` is interpreted as a single-channel sequence.
    """
    if x.dim() == 2:
        return x.unsqueeze(-1)
    if x.dim() == 3:
        return x
    raise ValueError(f"Expected a 2-D or 3-D input, got shape {tuple(x.shape)}")


class DNNClassifier(nn.Module):
    """Fully connected network on the flattened input.

    (batch, T, F) -> flatten -> (batch, T*F)
    -> [Linear -> BatchNorm -> ReLU -> Dropout] x len(hidden_dims)
    -> Linear -> (batch, n_classes)

    The flattening discards any notion of temporal order: every input
    position gets its own independent weights.
    """

    def __init__(
        self,
        sequence_length,
        n_features=1,
        n_classes=6,
        hidden_dims=(256, 128),
        dropout=0.3,
        batch_norm=True,
    ):
        super().__init__()
        self.input_dim = sequence_length * n_features

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
        x = x.reshape(x.size(0), -1)          # (B, T*F)
        if x.size(1) != self.input_dim:
            raise ValueError(
                f"DNN expects {self.input_dim} inputs per sample, got {x.size(1)}"
            )
        return self.classifier(self.features(x))


class _RecurrentClassifier(nn.Module):
    """Shared code for the vanilla RNN and the LSTM classifiers.

    (batch, T, F) -> recurrent layers -> (batch, T, H * directions)
    -> pooling over time -> (batch, H * directions)
    -> Dropout -> Linear -> (batch, n_classes)

    ``pooling`` selects the sequence summary:
      * "last": hidden state at the final time step (forward direction) and,
        if bidirectional, the backward state at t = 0;
      * "mean": average of all hidden states;
      * "max":  element-wise max of all hidden states.
    """

    rnn_cls = None

    def __init__(
        self,
        input_size=1,
        n_classes=6,
        hidden_size=64,
        num_layers=1,
        dropout=0.2,
        bidirectional=False,
        pooling="last",
        **rnn_kwargs,
    ):
        super().__init__()
        if pooling not in {"last", "mean", "max"}:
            raise ValueError(f"Unknown pooling '{pooling}'")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.pooling = pooling

        self.rnn = self.rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            **rnn_kwargs,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_dim, n_classes)

    def _pool(self, outputs):
        # outputs: (B, T, H * directions)
        if self.pooling == "mean":
            return outputs.mean(dim=1)
        if self.pooling == "max":
            return outputs.max(dim=1).values
        if self.bidirectional:
            forward_last = outputs[:, -1, : self.hidden_size]
            backward_last = outputs[:, 0, self.hidden_size :]
            return torch.cat([forward_last, backward_last], dim=1)
        return outputs[:, -1, :]

    def forward(self, x):
        x = _as_sequence(x)                   # (B, T, F)
        if x.size(-1) != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features per step, got {x.size(-1)}"
            )
        outputs, _ = self.rnn(x)              # (B, T, H * directions)
        summary = self._pool(outputs)         # (B, H * directions)
        return self.classifier(self.dropout(summary))


class RNNClassifier(_RecurrentClassifier):
    """Vanilla (Elman) RNN: h_t = tanh(W_ih x_t + W_hh h_{t-1} + b)."""

    rnn_cls = nn.RNN

    def __init__(self, *args, nonlinearity="tanh", **kwargs):
        super().__init__(*args, nonlinearity=nonlinearity, **kwargs)


class LSTMClassifier(_RecurrentClassifier):
    """LSTM: gated recurrence with a separate cell state."""

    rnn_cls = nn.LSTM


class CNNClassifier(nn.Module):
    """1-D convolutional network over the time axis.

    (batch, T, F) -> transpose -> (batch, F, T)
    -> [Conv1d -> BatchNorm -> ReLU -> MaxPool] x len(channels)
    -> global average pooling -> (batch, channels[-1])
    -> Dropout -> Linear -> (batch, n_classes)

    Kernels share weights along the time axis, so the network learns local
    patterns that can appear anywhere in the sequence.
    """

    def __init__(
        self,
        input_size=1,
        n_classes=6,
        channels=(32, 64, 128),
        kernel_size=5,
        pool_size=2,
        dropout=0.3,
    ):
        super().__init__()
        self.input_size = input_size

        blocks = []
        in_channels = input_size
        for out_channels in channels:
            blocks += [
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(pool_size),
            ]
            in_channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_channels, n_classes)

    def forward(self, x):
        x = _as_sequence(x).transpose(1, 2)   # (B, F, T)
        if x.size(1) != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} input channels, got {x.size(1)}"
            )
        x = self.features(x)                  # (B, C, T')
        x = self.global_pool(x).squeeze(-1)   # (B, C)
        return self.classifier(self.dropout(x))
