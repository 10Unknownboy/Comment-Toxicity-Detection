"""
Bidirectional LSTM model for multi-label toxicity classification.

Architecture
------------
Embedding → BiLSTM (stacked) → Dropout → FC → ReLU → Dropout → FC → Sigmoid

The final hidden states from both LSTM directions are concatenated and
passed through two fully-connected layers to produce independent
probabilities for each of the six toxicity labels.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ToxicityClassifier(nn.Module):
    """Multi-label toxicity classifier built on a Bidirectional LSTM.

    Parameters
    ----------
    vocab_size : int
        Total vocabulary size (including PAD and UNK tokens).
    embed_dim : int
        Dimensionality of the token-embedding vectors.
    hidden_dim : int
        Number of hidden units **per direction** in the LSTM.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout probability applied after the LSTM output and between
        the fully-connected layers.
    num_labels : int, optional
        Number of independent binary labels to predict (default ``6``).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_labels: int = 6,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_labels = num_labels

        # --- Embedding layer (PAD index = 0) ---
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,
        )

        # --- Bidirectional LSTM ---
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # --- Classifier head ---
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)  # *2 for bidirectional
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Integer-encoded input of shape ``(batch_size, seq_len)``.

        Returns
        -------
        torch.Tensor
            Raw logits of shape ``(batch_size, num_labels)``.
            Apply ``torch.sigmoid()`` externally to get probabilities.
        """
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        # LSTM output: (batch, seq_len, hidden*2)
        # h_n: (num_layers*2, batch, hidden)
        _, (h_n, _) = self.lstm(embedded)

        # Concatenate the final forward and backward hidden states from
        # the last LSTM layer.
        # h_n shape: (num_layers * 2, batch, hidden_dim)
        # Forward  → h_n[-2]  |  Backward → h_n[-1]
        forward_hidden = h_n[-2]   # (batch, hidden_dim)
        backward_hidden = h_n[-1]  # (batch, hidden_dim)
        hidden = torch.cat((forward_hidden, backward_hidden), dim=1)  # (batch, hidden*2)

        # Classifier
        out = self.dropout(hidden)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)  # (batch, num_labels) — raw logits

        return out
