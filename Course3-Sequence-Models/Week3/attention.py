import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProductAttention(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, masked=None):
        """
        Args:
            query: (batch_size, len_q, d_k)
            key: (batch_size, len_kv, d_k)
            value: (batch_size, len_kv, d_v)

        MatMul Q, K -> Scale -> Mask (Opt.) -> Softmax -> MatMul V
        Attention(Q, K, V) = softmax(Q*K^T/sqrt(d_k))*V

        Return:
            attention_weights: (batch_size, len_q, d_v)
        """
        # MatMul1
        d_k = query.size(-1)
        # Q, K prod: (batch_size, len_q, len_kv)
        prod = torch.einsum('nqd,nkd->nqk', query, key)

        # Scale
        scaled = prod / torch.sqrt(d_k)

        # Mask
        if masked is not None:
            scaled = scaled.masked_fill(masked==0, float('-1e20'))      # float('-inf')

        # Softmax
        self.attention_weights = F.softmax(scaled)

        # MatMul2
        # attention_weights shape: (batch_size, len_q, d_v)
        self.attention_weights = self.dropout(self.attention_weights)
        self.attention_weights = torch.einsum('nqk,nkd->nqd', self.attention_weights, value)

        return self.attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_head: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.h = num_head
        d_head = hidden_size // self.h

        assert d_head * self.h == self.d_model, 'Embedding size should be divided by h!'

        # Define W_q, W_k, W_v, and fc layer
        self.W_q = nn.LazyLinear(hidden_size, bias=False)
        self.W_k = nn.LazyLinear(hidden_size, bias=False)
        self.W_v = nn.LazyLinear(hidden_size, bias=False)
        self.fc = nn.LazyLinear(hidden_size, bias=False)
        self.attention = DotProductAttention(dropout=dropout)

    def forward(self, query, key, value, masked=None):
        # Multi-head Attention - Split embedding into h pieces
        query = self.transpose(self.W_q(query))
        key = self.transpose(self.W_k(key))
        value = self.transpose(self.W_v(value))

        output = self.attention(query, key, value, masked)
        output = self.transpose_concat(output)

        return self.fc(output)


    @staticmethod
    def transpose(self, x):
        # Input x: (batch_size, len_q(or len_kv), hidden_size)

        # (batch_size, h, len_q(or len_kv), hidden_size/h)
        x = x.view(x.size(0), self.h, x.size(1), -1)

        # (batch_size*h, len_q(or len_kv), hidden_size/h)
        x = x.view(-1, x.size(2), x.size(3))

        return x
    
    @staticmethod
    def transpose_concat(self, x):
        # Input x: (batch_size*h, len_q(or len_kv), hidden_size/h)

        # (batch_size, h, len_q(or len_kv), hidden_size/h)
        x = x.view(-1, self.h, x.size(1), x.size(2))

        # (batch_size, len_q(or len_kv), hidden_size)
        x = x.view(x.size(0), x.size(2), -1)

        return x
