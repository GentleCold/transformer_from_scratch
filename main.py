import argparse
import time

from torch import dropout

from train import Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", type=int, default=128, help="batch size, defaul=128"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=30, help="max number of epochs, default=30"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="learning rate, default=0.001",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout rate, default=0.5"
    )
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="optimizer type, default=adam"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="hidden dim, default=128"
    )
    parser.add_argument(
        "--feed_forward_dim",
        type=int,
        default=512,
        help="feed forward dim, default=512",
    )
    parser.add_argument(
        "--heads", type=int, default=8, help="heads of multi-head attention, default=8"
    )
    parser.add_argument(
        "--layers", type=int, default=2, help="layers of encoder and decoder, default=2"
    )

    args = parser.parse_args()
    model = Model(
        args.learning_rate,
        args.max_epochs,
        args.batch_size,
        args.hidden_dim,
        args.feed_forward_dim,
        args.layers,
        args.heads,
        args.dropout,
        args.optimizer,
    )

    start_time = time.time()
    model.train()
    end_time = time.time()

    print("Training time: ", end_time - start_time)
    model.metric(0, True)
    model.draw_loss()
