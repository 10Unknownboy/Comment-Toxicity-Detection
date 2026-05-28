"""
Bidirectional LSTM model for multi-label toxicity classification.

Architecture:
  Embedding -> BiLSTM (stacked) -> Dropout -> FC -> ReLU -> Dropout -> FC -> Sigmoid

The final hidden states from both LSTM directions are concatenated and
passed through two fully-connected layers to produce independent
probabilities for each of the six toxicity labels.
"""

import torch
import torch.nn as nn

import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for multi-label binary classification.
    
    Dynamically scales the loss based on prediction confidence to focus
    training on hard, rare examples while down-weighting easy negatives.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha  # Tensor of positive weights per label
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce_loss)  # probability of the true class
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss

        if self.alpha is not None:
            # Apply positive class weighting
            alpha_t = targets * self.alpha + (1 - targets) * 1.0
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

class ToxicityClassifier(nn.Module):
    """Multi-label toxicity classifier built on a Bidirectional LSTM.

    Constructor arguments:
      vocab_size  -- total vocabulary size (including PAD and UNK tokens)
      embed_dim   -- dimensionality of the token-embedding vectors
      hidden_dim  -- number of hidden units per direction in the LSTM
      num_layers  -- number of stacked LSTM layers
      dropout     -- dropout probability applied after the LSTM output
                     and between the fully-connected layers
      num_labels  -- number of independent binary labels to predict
                     (default 6)
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
        dropout,
        num_labels=6,
    ):
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

    def forward(self, x):
        """Forward pass.

        Takes an integer-encoded input tensor of shape (batch_size, seq_len).
        Returns raw logits of shape (batch_size, num_labels).  Apply
        torch.sigmoid() externally to get probabilities.
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
