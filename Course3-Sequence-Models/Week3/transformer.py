import torch
import torch.nn as nn

from attention import MultiHeadAttention
from utils import AddNorm, PositionWiseFFN


class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int, h: int, dropout: float):
        super().__init__()
        self.h = h
        self.dropout = dropout
        self.attention = MultiHeadAttention(hidden_size=hidden_size, num_head=h)


class TransformerEncoder(nn.Module):
    pass


class TransformerDecoderBlock(nn.Module):
    pass


class TransformerDecoder(nn.Module):
    pass
