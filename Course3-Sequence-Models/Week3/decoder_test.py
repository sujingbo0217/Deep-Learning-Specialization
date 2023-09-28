import torch

from transformer import TransformerEncoderBlock, TransformerDecoderBlock


X = torch.ones((2, 100, 24))
valid_lens = torch.tensor([3, 2])
encoder_blk = TransformerEncoderBlock(24, 48, 8, 0.5)
encoder_blk.eval()

encoder_shape = encoder_blk(X, valid_lens).shape
expected_shape = X.shape

assert encoder_shape == expected_shape, f'tensor\'s shape {encoder_shape} != expected shape {expected_shape}'
print('Same encoder shape!')

decoder_blk = TransformerDecoderBlock(hidden_size=24, ffn_hidden_size=48, num_head=8, dropout=0.5, i=0)
X = torch.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]

decoder_blk.train()
decoder_shape = decoder_blk(X, state)[0].shape
assert encoder_shape == expected_shape, f'tensor\'s shape {decoder_shape} != expected shape {expected_shape}'
print('Same decoder shape!')
