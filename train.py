import numpy as np
from torch import optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm

from data.data_handler import DEVICE, PAD_TOKEN, DataHandler
from model.lstm import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Train:
    def __init__(
        self,
        learning_rate=0.001,
        epochs=5,
        batch_size=128,
        enc_emb_dim=128,
        dec_emb_dim=128,
        hidden_dim=256,
        dropout=0.5,
        n_layers=1,
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
