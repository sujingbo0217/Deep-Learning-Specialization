# -*- coding = utf-8 -*-
# @Author : Jingbo Su
# @File : train.py

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

import spacy
import numpy as np

import math
import time
import random
# from tqdm import trange

import argparse
import logging
import os

import utils
from evaluate import evaluate
from seq2seq import Encoder, Decoder, Seq2Seq

SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help='Directory containing the dataset')
parser.add_argument('--save_dir', default='runs', help='Directory restoring model weights')
# parser.add_argument('--model_dir', default='./', help='Directory containing params.json')


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(model, iterator, optimizer, criterion, params) -> float:

    # set model to training mode
    model.train()

    """
    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    tt = trange(num_steps)

    for i in tt:
        # fetch the next training batch
        train_batch, labels_batch = next(data_iterator)

        # compute model output and loss
        output_batch = model(train_batch)
        loss = loss_fn(output_batch, labels_batch)

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()

        # Evaluate summaries only once in a while
        if i % params.save_summary_steps == 0:
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](output_batch, labels_batch) for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

        # update the average loss
        loss_avg.update(loss.item())
        tt.set_postfix(loss='{:05.3f}'.format(loss_avg()))

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    """

    epoch_loss = 0.

    for i, batch in enumerate(iterator):
        src = batch.src                                 # src_length, batch_size
        trg = batch.trg                                 # trg_length, batch_size

        outputs = model(src, trg)                       # trg_length, batch_size, output_size
        output_size = outputs.size(-1)

        y = trg[1:].view(-1)                            # get rid of "<sos>" && flatten
        y_pred = outputs[1:].view(-1, output_size)      # get rid of "<sos>" && flatten

        loss = criterion(y_pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params.grad_clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def train_and_evaluate(model, train_data, valid_data, optimizer, criterion, params, restore_file=None):

    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = f'{args.save_dir}.pt'
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    # Build vocabulary
    src.build_vocab(train_data, min_freq=2)
    trg.build_vocab(train_data, min_freq=2)

    print(f"Unique tokens in source (de) vocabulary: {len(src.vocab)}")
    print(f"Unique tokens in target (en) vocabulary: {len(trg.vocab)}")

    # Data iterator
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), batch_size=params.batch_size, device=params.device)

    best_valid_loss = float('inf')

    # Loop epochs
    for epoch in range(params.num_epochs):

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, params)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        elapsed_mins, elapsed_secs = epoch_time(start_time, end_time)

        is_best = valid_loss < best_valid_loss
        state = {
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
        }

        utils.save_checkpoint(state=state, is_best=is_best, checkpoint=args.save_dir)

        best_valid_loss = valid_loss
        torch.save(model.state_dict(), f'{args.save_dir}/best.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {elapsed_mins}m {elapsed_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    torch.save(model.state_dict(), f'{args.save_dir}/last.pt')

if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()

    json_path = './params.json'
    assert os.path.isfile(json_path), f'No json configuration file found at {json_path}'

    params = utils.Params(json_path)

    # use GPU if available
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the random seed for reproducible experiments
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if params.device == 'cuda':
        torch.cuda.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True

    # Set the logger
    utils.set_logger('./train.log')

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    """
    # load data
    data_loader = DataLoader(args.data_dir, params)
    data = data_loader.load_data(['train', 'val'], args.data_dir)
    train_data = data['train']
    val_data = data['val']
    
    # specify the train and val dataset sizes
    params.train_size = train_data['size']
    params.val_size = val_data['size']

    logging.info("- done.")
    """

    # python -m spacy download en_core_web_sm
    # python -m spacy download de_core_news_sm

    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')

    # Tokenize
    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    src = Field(tokenize=tokenize_de, lower=True, init_token='<sos>', eos_token='<eos>')
    trg = Field(tokenize=tokenize_en, lower=True, init_token='<sos>', eos_token='<eos>')

    # Load and split data
    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(src, trg))
    # train_data, valid_data, test_data = Multi30k(root='.data', split=('train', 'valid', 'test'), language_pair=('de', 'en'))

    logging.info("- done.")

    params.train_size = len(train_data.examples)
    params.valid_size = len(valid_data.examples)
    params.test_size = len(test_data.examples)

    logging.info(f"Number of training examples: {params.train_size}")
    logging.info(f"Number of validation examples: {params.valid_size}")
    logging.info(f"Number of testing examples: {params.test_size}")
    print(vars(train_data.examples[0]))

    # Define the model and optimizer
    encoder = Encoder(len(src.vocab), params.encoder_embedding_size, params.lstm_hidden_size, params.num_layers, params.encoder_dropout)
    decoder = Decoder(len(trg.vocab), params.decoder_embedding_size, params.lstm_hidden_size, params.num_layers, params.decoder_dropout)
    model = Seq2Seq(encoder, decoder, params.device).to(params.device)

    model.apply(init_weights)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=trg.vocab.stoi[trg.pad_token])

    # Train the model
    logging.info(f'Starting training for {params.num_epochs} epoch(s)')

    train_and_evaluate(model, train_data, valid_data, optimizer, criterion, params)
