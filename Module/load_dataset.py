# -*- coding = utf-8 -*-
# @Author : Jingbo Su
# @File : load_dataset.py

import os
import spacy    # For tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence     # To pad batch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
import pandas as pd
from PIL import Image

"""
We want to convert text -> numerical values
    1. We need a *Vocabulary* to map each word to an index
    2. We need to set up a PyTorch dataset to load the data
    3. Set up padding of every batch
        (all examples should be of the same *seq_len* and set up Dataloader)
"""

# Download with: python -m spacy download en_core_web_sm
spacy_eng = spacy.load("en_core_web_sm")


class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.stoi = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        assert len(self.itos) == len(self.stoi)
        return len(self.itos)

    # Use spacy to tokenize a sentence
    @staticmethod
    def tokenizer(text: str) -> list:
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    # Check if the frequency of a word is up to `freq_threshold`
    # Add the word to the dictionary, ignore it otherwise
    def build_vocabulary(self, sentence_list: list):
        i = 4
        freq_dic = {}

        for sentence in sentence_list:
            token_list = self.tokenizer(sentence)
            for token in token_list:
                if token not in freq_dic:
                    freq_dic[token] = 1
                else:
                    freq_dic[token] += 1

                if freq_dic.get(token) >= self.freq_threshold:
                    self.itos[i] = token
                    self.stoi[token] = i
                    i += 1

    # Tokenize first, return the token if it is in the stoi
    # Return <UNK> as unknown otherwise
    def numericalize(self, text: str) -> list:
        token_list = self.tokenizer(text)
        return [self.stoi.get(token) if token in self.stoi else self.stoi.get('<UNK>') for token in token_list]


class FlickrDataset(Dataset):
    def __init__(self, image_dir: str, captions_file: str, 
                 transform=None, freq_threshold=5):
        self.image_dir = image_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        # Get image, caption columns
        self.images = self.df.image
        self.captions = self.df.caption

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        caption = self.captions[item]
        image_id = self.images[item]
        image = Image.open(os.path.join(self.image_dir, image_id)).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        # Numericalized caption [<SOS>, numericalized_caption, <EOS>]
        numericalized_caption = [self.vocab.stoi['<SOS>']] + self.vocab.numericalize(caption) + [self.vocab.stoi['<EOS>']]

        return image, torch.tensor(numericalized_caption)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # batch: (image, caption)
        # return: (images, targets)
        images = [item[0].unsqueeze(0) for item in batch]   # 32*(1, 3, 224, 224)
        images = torch.cat(images, dim=0)                   # (32, 3, 224, 224)
        targets = [item[1] for item in batch]               # 32*(seq_len, )
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)  # padding: (max(len(targets)), 32)
        return images, targets

# Set up PyTorch DataLoader
def get_loader(root_dir: str, annotation_dir: str, transform=None, batch_size=32, num_workers=8, shuffle=True, pin_memory=True,):
    dataset = FlickrDataset(image_dir=root_dir, captions_file=annotation_dir, transform=transform)
    pad_idx = dataset.vocab.stoi.get('<PAD>')
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )
    return loader


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataloader = get_loader(root_dir='dataset/flickr8k/Images/', annotation_dir='dataset/flickr8k/captions.txt', transform=transform)
    for i, (images, captions) in enumerate(dataloader):
        print(images.shape, captions.shape)
