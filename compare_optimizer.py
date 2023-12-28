import time

from matplotlib import pyplot as plt

from train import Model

if __name__ == "__main__":
    model_adam = Model(optimizer="adam")
    model_adamw = Model(optimizer="adamw")
    model_sgd = Model(optimizer="sgd")

    print("===== Adam =====")
    start_time = time.time()
    model_adam.train(verbose=False)
    end_time = time.time()
    print("Training time: ", end_time - start_time)
    model_adam.metric(0)

    print("\n===== Adamw =====")
    start_time = time.time()
    model_adamw.train(verbose=False)
    end_time = time.time()
    print("Training time: ", end_time - start_time)
    model_adamw.metric(0)

    print("\n===== Sgd =====")
    start_time = time.time()
    model_sgd.train(verbose=False)
    end_time = time.time()
    print("Training time: ", end_time - start_time)
    model_sgd.metric(0)

    # show plt
    plt.plot(model_adam.train_loss, label="Adam")
    plt.plot(model_adamw.train_loss, label="Adamw")
    plt.plot(model_sgd.train_loss, label="sgd")
    plt.title("Smoothed Train Loss")
    plt.legend()
    plt.show()
