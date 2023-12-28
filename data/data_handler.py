"""
Author: GentleCold
Reference:
1. https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
2. https://pytorch.org/tutorials/beginner/translation_transformer.html#collation

[1] for handling dataset from scratch
[2] for add padding per batch
"""
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Vocabulary:
    """
    Make map and build vocabulary, using integer encoding
    """

    def __init__(self):
        self.word2index = {}
        self.index2word = {
            PAD_TOKEN: "PAD",
            SOS_TOKEN: "SOS",
            EOS_TOKEN: "EOS",
            UNK_TOKEN: "UNK",
        }
        self.nums = 4
        # self.max_length = 0

    def add_sentence(self, sentence):
        list_sentence = sentence.split(" ")
        # if len(list_sentence) > self.max_length:
        #     self.max_length = len(list_sentence) + 2  # including <SOS> and <EOS>
        for word in list_sentence:
            self._add_word(word)

    def sentence2index(self, sentence):
        return [
            self.word2index[word] if word in self.word2index else UNK_TOKEN
            for word in sentence.split(" ")
        ]

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.nums
            self.index2word[self.nums] = word
            self.nums += 1


class MyIterableDataset(IterableDataset):
    """
    To build dynamic length of dataset
    """

    def __init__(self, dataset, source_voc, target_voc):
        self.dataset = dataset
        self.source_voc = source_voc
        self.target_voc = target_voc

    def __iter__(self):
        for _, sample in self.dataset.iterrows():
            one_source_idx = self.source_voc.sentence2index(sample["description"])
            one_target_idx = self.target_voc.sentence2index(sample["diagnosis"])

            one_source_idx.insert(0, SOS_TOKEN)
            one_source_idx.append(EOS_TOKEN)

            one_target_idx.insert(0, SOS_TOKEN)
            one_target_idx.append(EOS_TOKEN)

            yield one_source_idx, one_target_idx

    def __len__(self):
        return len(self.dataset)


class DataHandler:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        raw_train_dataset = (
            pd.read_csv("./data/train.csv")
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
        self.test_dataset = pd.read_csv("./data/test.csv")

        # Split data
        train_ratio = 0.8
        train_size = int(len(raw_train_dataset) * train_ratio)

        self.train_dataset = raw_train_dataset.iloc[:train_size]
        self.val_dataset = raw_train_dataset.iloc[train_size:]

        # Build vocabulary
        self.source_voc = Vocabulary()
        self.target_voc = Vocabulary()

        for sentence in self.train_dataset["description"]:
            self.source_voc.add_sentence(sentence)
        for sentence in self.train_dataset["diagnosis"]:
            self.target_voc.add_sentence(sentence)

    def get_dataloader(self):
        return (
            self._get_dataloader(self.train_dataset),
            self._get_dataloader(self.val_dataset),
            self._get_dataloader(self.test_dataset),
        )

    def _get_dataloader(self, dataset):
        """
        Transform dataset to dataloader in pytorch
        """
        iterable_dataset = MyIterableDataset(dataset, self.source_voc, self.target_voc)

        loader = DataLoader(
            iterable_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
        )
        return loader

    @staticmethod
    def _collate_fn(batch):
        """
        Used to padding according to the batch samples, which will decrease the inputdim
        """
        max_source_len = max(len(sample[0]) for sample in batch)
        max_target_len = max(len(sample[1]) for sample in batch)

        padded_batch = []
        for source_idx, target_idx in batch:
            padded_source_idx = source_idx + [PAD_TOKEN] * (
                max_source_len - len(source_idx)
            )
            padded_target_idx = target_idx + [PAD_TOKEN] * (
                max_target_len - len(target_idx)
            )

            padded_batch.append((padded_source_idx, padded_target_idx))

        source_tensor = torch.LongTensor([sample[0] for sample in padded_batch]).to(
            DEVICE
        )
        target_tensor = torch.LongTensor([sample[1] for sample in padded_batch]).to(
            DEVICE
        )

        return source_tensor, target_tensor
