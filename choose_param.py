import json

from tqdm import tqdm

from train import Model


def explore_hyperparameters(param_grid, current_params={}, all_params=[]):
    if not param_grid:
        all_params.append(current_params)
        return {}

    current_param, remaining_params = list(param_grid.items())[0]
    for value in remaining_params:
        updated_params = current_params.copy()
        updated_params[current_param] = value
        explore_hyperparameters(
            dict(list(param_grid.items())[1:]), updated_params, all_params
        )

    return all_params


param_grid = {
    "learning_rate": [0.001, 0.0005, 0.0001],
    "hidden_dim": [128, 256],
    "ff_dim": [512, 1024],
    "n_layers": [2, 4],
    "dropout": [0.2, 0.5],
}

all_params = explore_hyperparameters(param_grid)

results = []
for params in tqdm(all_params):
    model = Model(**params)
    valid_loss, train_epoch = model.train(verbose=False)
    bleu, meteor, rouge_l, rouge_1, rouge_2 = model.metric(0)
    metrics = {
        "valid_loss": valid_loss,
        "bleu": bleu,
        "meteor": meteor,
        "rouge_l": rouge_l,
        "rouge_1": rouge_1,
        "rouge_2": rouge_2,
    }
    params["epoches"] = train_epoch
    results.append([params, metrics])

# sort according to valid_loss
results = sorted(results, key=lambda x: x[1]["valid_loss"])
best_params = results[0][0]
best_metrics = results[0][1]

print("The best params is:")
print(f"learning_rate: {best_params['learning_rate']}")
print(f"epoches: {best_params['epoches']}")
print(f"hidden_dim: {best_params['hidden_dim']}")
print(f"ff_dim: {best_params['ff_dim']}")
print(f"n_layers: {best_params['n_layers']}")
print(f"dropout: {best_params['dropout']}")

print("\nThe best metrics is:")
print(f"valid_loss: {best_metrics['valid_loss']}")
print(f"bleu: {best_metrics['bleu']}")
print(f"meteor: {best_metrics['meteor']}")
print(f"rouge_l: {best_metrics['rouge_l']}")
print(f"rouge_1: {best_metrics['rouge_1']}")
print(f"rouge_2: {best_metrics['rouge_2']}")

# save total results
with open("output/choose_param.json", "w") as file:
    json.dump(results, file, indent=2, separators=(", ", ": "))
