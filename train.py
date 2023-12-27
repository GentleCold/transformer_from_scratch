import numpy as np
from torch import optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torchtext.data.metrics import bleu_score
from tqdm import tqdm

from data.data_handler import DEVICE, EOS_TOKEN, PAD_TOKEN, SOS_TOKEN, DataHandler
from model.lstm import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Model:
    def __init__(
        self,
        learning_rate=0.001,
        epochs=5,
        batch_size=128,
        enc_emb_dim=128,
        dec_emb_dim=128,
        hidden_dim=256,
        dropout=0.5,
        n_layers=2,
    ):
        set_seed(42)
        self.epochs = epochs
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
            enc_emb_dim,
            hidden_dim,
            n_layers,
            dropout,
        )
        self.dec = Decoder(
            self.data_handler.target_voc.nums,
            dec_emb_dim,
            hidden_dim,
            n_layers,
            dropout,
        )
        self.model = Seq2Seq(self.enc, self.dec, DEVICE).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # ignore padding token
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    def train(self):
        print("===== Traning Info =====")
        print("Device:", DEVICE)
        print("Batch size:", self.batch_size)
        print(f"Model Info:\n")
        print(self.model)

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"The model has {count_parameters(self.model):,} trainable parameters")
        print("\n==== Starting Train ====")

        for epoch in range(1, self.epochs + 1):
            self._epoch_train(epoch)
            self._evaluate(self.val_loader)

        self._evaluate(self.test_loader)

    def _epoch_train(self, epoch):
        self.model.train()
        epoch_loss = 0
        for source, target in tqdm(self.train_loader):
            self.optimizer.zero_grad()
            output = self.model(source, target)

            # trim the first token
            output = output[1:].view(-1, output.shape[-1])
            target = target[1:].reshape(-1)

            loss = self.criterion(output, target)
            loss.backward()

            # used to prevent gradient explosion
            clip_grad_norm_(self.model.parameters(), 1)

            self.optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(self.train_loader)
        print(f"Train Epoch {epoch}")
        print("Train set: \nLoss: {}".format(epoch_loss))

    def _evaluate(self, dataset):
        self.model.eval()
        epoch_loss = 0

        with torch.no_grad():
            for source, target in tqdm(dataset):
                self.optimizer.zero_grad()
                output = self.model(
                    source, target, 0
                )  # turn off teacher forcing with zero

                # trim the first token
                output = output[1:].view(-1, output.shape[-1])
                target = target[1:].reshape(-1)

                loss = self.criterion(output, target)

                epoch_loss += loss.item()

        epoch_loss /= len(dataset)
        if dataset == self.val_loader:
            print(f"Val set: \nLoss: {epoch_loss}\n")
        else:
            print(f"Test set: \nLoss: {epoch_loss}\n")

    def _inference(self, sentence, max_len):
        self.model.eval()

        source_index = self.data_handler.source_voc.sentence2index(sentence)
        source_index.insert(0, SOS_TOKEN)
        source_index.append(EOS_TOKEN)
        source_tensor = torch.LongTensor(source_index).unsqueeze(1).to(DEVICE)
        predicts = [SOS_TOKEN]

        with torch.no_grad():
            hidden, cell = self.model.encoder(source_tensor)

            # first input to the decoder is the <sos> tokens
            input = source_tensor[0, :]

            for _ in range(1, max_len):
                with torch.no_grad():
                    output, hidden, cell = self.model.decoder(input, hidden, cell)

                top1 = output.argmax(1)
                input = top1
                predict = input.item()
                predicts.append(input.item())
                if predict == EOS_TOKEN:
                    break
        predicts = [self.data_handler.target_voc.index2word[idx] for idx in predicts]
        return predicts[1:]

    def bleu(self):
        targets = []
        predicts = []
        for _, sample in tqdm(self.data_handler.test_dataset.iterrows()):
            targets.append(sample["diagnosis"].split(" "))
            predict = self._inference(
                sample["description"], max_len=len(sample["diagnosis"].split(" "))
            )
            predicts.append(predict[:-1])
        print(f"BLEU: {bleu_score(predicts, targets)}")
        print(targets, predicts)
