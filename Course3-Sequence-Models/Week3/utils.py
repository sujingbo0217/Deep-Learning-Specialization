import torch
import torch.nn as nn

import time
import numpy as np


class FeedForward(nn.Module):
    def __init__(self, ffn_hidden_size: int, ffn_output_size: int):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_hidden_size)    # (hidden_size, ffn_hidden_size)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_output_size)    # (ffn_hidden_size, hidden_size)

    def forward(self, x):
        return self.dense2(self.relu(self.dense1(x)))


class AddNorm(nn.Module):
    def __init__(self, norm_shape: int, dropout: float) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(norm_shape)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, Y):
        """
        Args:
            X: Output of Multi-head Attention or Feed Forward
            Y: Residual input
        """
        return self.dropout(self.layer_norm(X + Y))


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int, dropout: float, max_len=1000):
        super().__init__()
        assert hidden_size % 2 == 0, 'Odd hidden_size cannot use sin/cos positional encoding'
        
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len
        
        """
        PE(pos, 2i) = sin(pos/1000^{2i/d_model})
        PE(pos, 2i+1) = cos(pos/1000^{2i/d_model})
        """
        self.PE = torch.zeros(1, max_len, hidden_size)
        X = torch.arange(0, max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, hidden_size, 2, dtype=torch.float32)/hidden_size)
        
        self.PE[:, :, 0::2] = torch.sin(X)
        self.PE[:, :, 1::2] = torch.cos(X)

    def forward(self, x):
        # x shape (batch_size, num_step, hidden_size)
        x = x + self.PE[:, :max(x.size(1), self.max_len), :].to(x.device)
        return self.dropout(x)


class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`sec_minibatch_sgd`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()