import copy
import optuna

from configs import config
import train


def objective(trial):

    c = copy.deepcopy(config)

    c.lr = trial.suggest_float(
        "lr",
        1e-6,
        1e-4,
        log=True
    )

    c.batch_size = trial.suggest_categorical(
        "batch_size",
        [40, 48, 56]
    )

    c.weight_decay = trial.suggest_float(
        "weight_decay",
        1e-7,
        1e-5,
        log=True
    )

    c.epochs = 100
    c.save_epochs = c.epochs

    accuracy = train.main(c)

    return accuracy


study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=30)

print(study.best_params)
print(study.best_value)