import torch

from transformer import TransformerEncoder

encoder = TransformerEncoder(src_vocab_size=200, hidden_size=24, ffn_hidden_size=48, num_head=8, num_block=2, dropout=0.5)
x = torch.ones((2, 100), dtype=torch.long)

encoder_shape = encoder(x).shape
expected_shape = (2, 100, 24)

assert encoder_shape == expected_shape, f'tensor\'s shape {encoder_shape} != expected shape {expected_shape}'
print('Same shape!')
