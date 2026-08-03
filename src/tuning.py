import copy
import optuna

from configs import config
import train


def objective(trial):

    c = copy.deepcopy(config)

    c.lr = trial.suggest_float(
        "lr",
        1e-5,
        6e-5,
        log=True
    )

    c.batch_size = trial.suggest_categorical(
        "batch_size",
        [16, 32, 48]
    )

    c.weight_decay = trial.suggest_float(
        "weight_decay",
        1e-6,
        5e-6,
        log=True
    )

    c.epochs = 100
    c.save_epochs = c.epochs

    accuracy = train.main(c, trial=trial)

    return accuracy

pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,    # Don't prune the first 5 trials
    n_warmup_steps=10       # Don't prune before epoch 10
)

study = optuna.create_study(
    direction="maximize",
    pruner=pruner
)

study.optimize(objective, n_trials=31)

print(study.best_params)
print(study.best_value)