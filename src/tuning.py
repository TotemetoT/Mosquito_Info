import copy
import optuna

from configs import config
import train


def objective(trial):

    c = copy.deepcopy(config)

    c.lr = trial.suggest_float(
        "lr",
        1e-4,
        1e-3,
        log=True
    )

    c.batch_size = trial.suggest_categorical(
        "batch_size",
        [32, 64, 128]
    )

    c.epochs = 50

    accuracy = train.main(c)

    return accuracy


study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=30)

print(study.best_params)
print(study.best_value)