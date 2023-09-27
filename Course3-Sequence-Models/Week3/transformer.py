import torch
import torch.nn as nn

import math

from attention import MultiHeadAttention
from utils import AddNorm, FeedForward, PositionalEncoding


class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int, num_head: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size=hidden_size, num_head=num_head, dropout=dropout)
        self.addnorm1 = AddNorm(norm_shape=hidden_size, dropout=dropout)
        self.ffn = FeedForward(ffn_hidden_size=ffn_hidden_size, ffn_output_size=hidden_size)
        self.addnorm2 = AddNorm(norm_shape=hidden_size, dropout=dropout)

    def forward(self, x, masked=None):
        x1 = x
        y1 = self.attention(query=x, key=x, value=x, masked=masked)
        x2 = self.addnorm1(y1, x1)
        y2 = self.ffn(x2)

        return self.addnorm2(y2, x2)


class TransformerEncoder(nn.Module):
    def _init__(self, src_vocab_size: int, hidden_size: int, ffn_hidden_size: int, num_head: int, num_block: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(src_vocab_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size, dropout, )
        self.blocks = nn.Sequential()
        
        for i in range(num_block):
            self.blocks.add_module(f'EncoderBlock{i}', TransformerEncoderBlock(hidden_size, ffn_hidden_size, num_head, dropout))

    def forward(self, x, masked=None):
        # Embedding
        # In the embedding layers, we multiply those weights by sqrt(d_model)
        x = self.embedding(x) * math.sqrt(self.hidden_size)
        
        # Position Encoding
        x = self.pos_encoding(x)

        self.attention_weights = [] * len(self.blocks)

        for i, block in enumerate(self.blocks):
            x = block(x, masked)
            self.attention_weights[i] = block.attention.attention.attention_weights

        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int, num_head: int, dropout: float, i: int):
        super().__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(hidden_size, num_head, dropout)
        self.addnorm1 = AddNorm(hidden_size, dropout)
        self.attention2 = MultiHeadAttention(hidden_size, num_head, dropout)
        self.addnorm2 = AddNorm(hidden_size, dropout)
        self.ffn = FeedForward(ffn_hidden_size, hidden_size)
        self.addnorm3 = AddNorm(hidden_size, dropout)

    def forward(self, x, state):
        """
        state[0]: encoder output
        state[1]: len=num_block, state[2][self.i] contains representations of the decoded output at the i-th block up to the current time step
        """
        encoder_output = state[0]

        if state[1][self.i] is None:
            # Output from encoder
            key_value = x
        else:
            # Output from previous decoder
            key_value = torch.concat((state[2][self.i], x), dim=1)
        state[1][self.i] = key_value

        if self.training:
            batch_size, num_step, _ = x.shape
            masked = torch.arange(1, num_step+1, device=x.device).repeat(batch_size, 1)     # (batch_size, num_step)
        else:
            masked = None

        # Masked Multi-head Attention
        x1 = self.attention1(query=x, key=key_value, value=key_value, masked=masked)
        y = self.addnorm1(x1, x)

        # Encoder output shape (batch_size, num_step, num_hidden)
        x2 = self.attention2(query=y, query=encoder_output, key=encoder_output, masked=None)
        y1 = self.addnorm2(x2, y)

        # Feed Forward Network
        x3 = self.ffn(y1)
        y2 = self.addnorm3(x3, y1)

        return y2, state


class TransformerDecoder(nn.Module):
    def __init__(self, trg_vocab_size: int, hidden_size: int, ffn_hidden_size: int, num_head: int, num_block: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(trg_vocab_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size, dropout, )
        self.blocks = nn.Sequential()

        for i in range(num_block):
            self.blocks.add_module(f'DecoderBlock{i}', TransformerDecoderBlock(hidden_size, ffn_hidden_size, num_head, dropout, i))

        self.dense = nn.LazyLinear(trg_vocab_size)

    def forward(self, x, state):
        x = self.pos_encoding(self.embedding(x) * math.sqrt(self.hidden_size))
        self._attention_weights = [[] * len(self.blocks) for _ in range(2)]
        
        for i, block in enumerate(self.blocks):
            x = block(x, state)
            # Decoder self-attention (with mask) weights
            self._attention_weights[0][i] = block.attention1.attention.attention_weights
            # Encoder-Decoder self-attention weights
            self._attention_weights[1][i] = block.attention2.attention.attention_weights

        return self.dense(x), state
    
    @property
    def attention_weights(self):
        return self._attention_weights
