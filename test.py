"""
This 'all in one' file is written to learn the process.
I'll refactor it later.
"""
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.cuda import init
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm

from model.lstm import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 10

# Handle data

PAD_TOKEN = 0
EOS_TOKEN = 1
UNK_TOKEN = 2


class Vocabulary:
    """
    Make map and build vocabulary, using integer encoding
    """

    def __init__(self):
        self.word2index = {}
        self.index2word = {0: "PAD", 1: "EOS", 2: "UNK"}
        self.nums = 3
        self.max_length = 0

    def add_sentence(self, sentence):
        list_sentence = sentence.split(" ")
        if len(list_sentence) > self.max_length:
            self.max_length = len(list_sentence) + 1  # including <EOS>
        for word in list_sentence:
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.nums
            self.index2word[self.nums] = word
            self.nums += 1


# Split dataset
raw_dataset = (
    pd.read_csv("./data/train.csv")
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

train_ratio = 0.8

train_size = int(len(raw_dataset) * train_ratio)

train_dataset = raw_dataset.iloc[:train_size]
val_dataset = raw_dataset.iloc[train_size:]

# Build vocabulary
input_voc = Vocabulary()
output_voc = Vocabulary()

for sentence in train_dataset["description"]:
    input_voc.add_sentence(sentence)
for sentence in train_dataset["diagnosis"]:
    output_voc.add_sentence(sentence)

# Prepare training data
input_idx = np.zeros((train_size, input_voc.max_length), dtype=np.float32)
output_idx = np.zeros((train_size, input_voc.max_length), dtype=np.float32)


def Sentence2Index(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(" ")]


for i, sample in train_dataset.iterrows():
    in_ids = Sentence2Index(input_voc, sample["description"])
    ou_ids = Sentence2Index(output_voc, sample["diagnosis"])
    in_ids.append(EOS_TOKEN)
    ou_ids.append(EOS_TOKEN)
    input_idx[i, : len(in_ids)] = in_ids
    output_idx[i, : len(ou_ids)] = ou_ids

tensor_train_dataset = TensorDataset(
    torch.LongTensor(input_idx).to(DEVICE), torch.LongTensor(output_idx).to(DEVICE)
)
loader = DataLoader(
    tensor_train_dataset,
    batch_size=BATCH_SIZE,
)

# Train model
print(input_voc.nums)
print(input_voc.max_length)
print(output_voc.nums)
print(output_voc.max_length)

INPUT_DIM = input_voc.nums
OUTPUT_DIM = output_voc.nums
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for input, output in tqdm(loader):
        optimizer.zero_grad()
        outputs = model(input, output, 0)

        output_dim = outputs.shape[-1]

        outputs = outputs[1:].view(-1, output_dim)
        trg = output[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(outputs, trg)
        train_loss += loss

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: Train Loss={train_loss/len(loader):.4f}")
