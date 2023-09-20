# -*- coding = utf-8 -*-
# @Author : Jingbo Su
# @File : train.py

import torch
import torch.nn as nn
import random


class Encoder(nn.Module):
    def __init__(self, input_size: int, embedding_size: int, hidden_size: int, num_layers: int, dropout=0.2):
        """
        Args:
            input_size: original length of input sequence
            embedding_size: dimension after embedding layer
            hidden_size: dimension of hidden and cell state (horizontal)
            num_layers: number of layers (vertical)
            dropout: prevent overfitting
        """

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.net = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

    def forward(self, src):
        """
        Args:
            src: src_length * batch_size
        """

        x = self.dropout(self.embedding(src))
        output, hidden, cell = self.net(x)

        """
        x: src_length, batch_size, embedding_size
        output: src_length, batch_size, hidden_size
        hidden, cell: num_layers, batch_size, hidden_size
        """

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_size: int, embedding_size: int, hidden_size: int, num_layers: int, dropout=0.2):
        """
        Args:
            output_size: result/target length of output
            embedding_size: dimension after embedding layer
            hidden_size: dimension of hidden and cell state (horizontal)
            num_layers: number of layers (vertical)
            dropout: prevent overfitting
        """

        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.net = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

    def forward(self, token, hidden, cell):
        """
        Args:
            token: previous single word predicted by decoder        (batch_size) -> (1, batch_size)
            hidden: (previous) hidden state from encoder output     (num_layers, batch_size, hidden_size)
            cell: (previous) cell state from encoder output         (num_layers, batch_size, hidden_size)
        """

        token = token.unsqueeze(0)  # (N) -> (1, N)
        x = self.dropout(self.embedding(token))
        output, hidden, cell = self.net(x, (hidden, cell))
        # output: (seq_len=1, batch_size, hidden_size) -> (*, hidden_size)
        pred = self.fc_out(output.squeeze(0))
        # pred: (batch_size, output_size)

        return pred, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hidden_size == decoder.hidden_size, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.num_layers == decoder.num_layers, "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Args:
            src: src_length, batch_size
            trg: trg_length, batch_size
            teacher_forcing_ratio: ratio of using ground_truth/(ground_truth+pred)
        """

        assert src.size(1) == trg.size(1)

        batch_size = src.size(1)
        trg_length = trg.size(0)
        output_size = self.decoder.output_size

        outputs = torch.zeros(trg_length, batch_size, output_size).to(self.device)

        hidden, cell = self.encoder(src)
        c0 = trg[0, :]  # decoder first input '<sos>'

        for tt in range(1, trg_length):
            pred, hidden, cell = self.decoder(c0, hidden, cell)
            outputs[tt] = pred
            teacher_forcing = random.random() < teacher_forcing_ratio
            c0 = trg[tt] if teacher_forcing else pred.argmax(1)

        return outputs
