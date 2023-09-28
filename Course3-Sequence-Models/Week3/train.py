import d2l.torch as d2l

from utils import Timer
from transformer import TransformerEncoder, TransformerDecoder


data = d2l.MTFraEng(batch_size=128)
hidden_size, num_block, dropout = 256, 2, 0.2
ffn_hidden_size, num_head = 64, 4
learning_rate = 0.01

encoder = TransformerEncoder(len(data.src_vocab), hidden_size, ffn_hidden_size, num_head, num_block, dropout)
decoder = TransformerDecoder(len(data.tgt_vocab), hidden_size, ffn_hidden_size, num_head, num_block, dropout)
model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'], lr=learning_rate)

trainer = d2l.Trainer(max_epochs=1, gradient_clip_val=1, num_gpus=0)

timer = Timer()
trainer.fit(model, data)
timer.stop()

print(f'total time: {timer.sum():.2f} s')
