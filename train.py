import math
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from torch import optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torchtext.data.metrics import bleu_score
from tqdm import tqdm

from data.data_handler import DEVICE, EOS_TOKEN, PAD_TOKEN, SOS_TOKEN, DataHandler

# from model.lstm import *
from model.transformer import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Model:
    def __init__(
        self,
        learning_rate=0.001,
        max_epochs=30,
        batch_size=128,
        hidden_dim=128,
        ff_dim=512,
        n_layers=2,
        heads=8,
        dropout=0.2,
        optimizer="adam",
    ):
        set_seed(42)
        self.epochs = max_epochs
        self.batch_size = batch_size

        self.data_handler = DataHandler(batch_size)
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
        ) = self.data_handler.get_dataloader()

        # Record loss
        # self.train_loss = []
        # self.val_loss = []
        #
        # self.train_accuracy = []
        # self.val_accuracy = []
        # self.test_result = None

        # Define model
        self.enc = Encoder(
            self.data_handler.source_voc.nums,
            hidden_dim,
            n_layers,
            heads,
            ff_dim,
            dropout,
        )
        self.dec = Decoder(
            self.data_handler.target_voc.nums,
            hidden_dim,
            n_layers,
            heads,
            ff_dim,
            dropout,
        )
        self.model = Seq2Seq(self.enc, self.dec).to(DEVICE)

        if optimizer == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        elif optimizer == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        # ignore padding token
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

        def initialize_weights(m):
            if hasattr(m, "weight") and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)

        self.model.apply(initialize_weights)

    def train(self, verbose=True):
        self.train_loss = []
        self.val_loss = []

        print("===== Traning Info =====")
        print("Device:", DEVICE)
        if verbose:
            print("Batch size:", self.batch_size)
            print(f"Model Info:\n")
            print(self.model)

            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(
                f"The model has {count_parameters(self.model):,} trainable parameters"
            )
        print("\n==== Starting Train ====")

        best_valid_loss = float("inf")
        early_stop_patience = 3
        early_stop_count = 0
        epoch = 0

        for epoch in range(1, self.epochs + 1):
            self._epoch_train(epoch, verbose)
            valid_loss = self._evaluate(self.val_loader, verbose)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), "output/model.pt")
                early_stop_count = 0
            else:
                early_stop_count += 1
                if early_stop_count == early_stop_patience:
                    print("Early stop...")
                    break

        print(f"The training epoch is {epoch}")
        print(f"Choose model with best valid loss: {best_valid_loss}")
        self.model.load_state_dict(torch.load("model.pt"))
        self._evaluate(self.test_loader, True)

        def moving_average(data, window_size):
            weights = np.repeat(1.0, window_size) / window_size
            smoothed_data = np.convolve(data, weights, "valid")
            return smoothed_data

        window_size = 10
        self.train_loss = moving_average(self.train_loss, window_size)
        self.val_loss = moving_average(self.val_loss, window_size)
        return best_valid_loss, epoch

    def _epoch_train(self, epoch, verbose):
        self.model.train()
        epoch_loss = 0
        for source, target in tqdm(self.train_loader):
            self.optimizer.zero_grad()
            # trim the last token
            output, _ = self.model(source, target[:, :-1])

            output = output.contiguous().view(-1, output.shape[-1])
            # trim the first token
            target = target[:, 1:].contiguous().view(-1)

            loss = self.criterion(output, target)
            loss.backward()

            # used to prevent gradient explosion
            clip_grad_norm_(self.model.parameters(), 1)

            self.optimizer.step()

            self.train_loss.append(loss.item())
            epoch_loss += loss.item()

        epoch_loss /= len(self.train_loader)
        if verbose:
            print(f"Train Epoch {epoch}")
            print(
                "Train set: \nLoss: {} PPL: {}".format(epoch_loss, math.exp(epoch_loss))
            )

    def _evaluate(self, dataset, verbose):
        self.model.eval()
        epoch_loss = 0

        with torch.no_grad():
            for source, target in dataset:
                self.optimizer.zero_grad()
                output, _ = self.model(source, target[:, :-1])

                output = output.contiguous().view(-1, output.shape[-1])
                # trim the first token
                target = target[:, 1:].contiguous().view(-1)

                loss = self.criterion(output, target)

                if dataset == self.val_loader:
                    self.val_loss.append(loss.item())
                epoch_loss += loss.item()

        epoch_loss /= len(dataset)
        if dataset == self.val_loader:
            if verbose:
                print(f"Val set: \nLoss: {epoch_loss} PPL: {math.exp(epoch_loss)}\n")
        else:
            print(f"Test set: \nLoss: {epoch_loss} PPL: {math.exp(epoch_loss)}\n")
        return epoch_loss

    def _inference(self, sentence, max_len):
        self.model.eval()

        source_index = self.data_handler.source_voc.sentence2index(sentence)
        source_index.insert(0, SOS_TOKEN)
        source_index.append(EOS_TOKEN)

        source_tensor = torch.LongTensor(source_index).unsqueeze(0).to(DEVICE)
        source_mask = self.model.make_src_mask(source_tensor)

        predicts = [SOS_TOKEN]

        # get contextual information
        with torch.no_grad():
            enc_src = self.model.encoder(source_tensor, source_mask)

        for _ in range(max_len):
            target_tensor = torch.LongTensor(predicts).unsqueeze(0).to(DEVICE)
            target_mask = self.model.make_trg_mask(target_tensor)

            with torch.no_grad():
                output, _ = self.model.decoder(
                    target_tensor, enc_src, target_mask, source_mask
                )
            pred_token = output.argmax(2)[:, -1].item()
            predicts.append(pred_token)
            if pred_token == EOS_TOKEN:
                break
        predicts = [self.data_handler.target_voc.index2word[idx] for idx in predicts]
        return predicts

    def metric(self, head):
        print("\n==== Calculating metrics ====")
        reference = []
        candidate = []
        sources = self.data_handler.test_dataset["description"].tolist()
        targets = self.data_handler.test_dataset["diagnosis"].tolist()
        rouge = Rouge()
        rouge_l_f = 0
        rouge_1_f = 0
        rouge_2_f = 0
        meteor = 0

        for i in tqdm(range(len(sources))):
            reference.append([targets[i].split(" ")])
            predict = self._inference(
                sources[i], max_len=len(targets[i].split(" ")) + 10
            )
            candidate.append(predict[1:-1])  # cut SOS_TOKEN and EOS_TOKEN

            rouge_score = rouge.get_scores(" ".join(predict[1:-1]), targets[i])[0]
            rouge_l_f += rouge_score["rouge-l"]["f"]  # type: ignore
            rouge_1_f += rouge_score["rouge-1"]["f"]  # type: ignore
            rouge_2_f += rouge_score["rouge-2"]["f"]  # type: ignore
            meteor += meteor_score(reference[-1], candidate[-1])

            if i < head:
                print(f"Reference: {reference[-1]}")
                print(f"Predict: {predict}")

        bleu = bleu_score(candidate, reference)
        meteor /= len(sources)
        rouge_l_f /= len(sources)
        rouge_1_f /= len(sources)
        rouge_2_f /= len(sources)

        print(f"BLEU: {bleu}")
        print(f"Meteor: {meteor}")
        print(f"ROUGE-L f1: {rouge_l_f}")
        print(f"ROUGE-1 f1: {rouge_1_f}")
        print(f"ROUGE-2 f1: {rouge_2_f}")
        return bleu, meteor, rouge_l_f, rouge_1_f, rouge_2_f

    def draw_loss(self):
        plt.plot(self.train_loss, label="Smoothed train Loss")
        plt.plot(self.val_loss, label="Smoothed val Loss")
        plt.title("Loss")
        plt.legend()
        plt.show()
