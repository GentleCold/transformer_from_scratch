import time

from train import Train

if __name__ == "__main__":
    train = Train()

    start_time = time.time()
    train.train()
    end_time = time.time()

    print("Training time: ", end_time - start_time)
