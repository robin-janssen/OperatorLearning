# This script implements an OPTUNA study to optimize the hyperparameters of a MultiONet for the branca dataset.


# Import necessary libraries
import optuna
import numpy as np
from functools import partial

from data import create_dataloader_chemicals
from training import BChemicalTrainConfig, train_multionet_chemical
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_intermediate_values,
)


# Define the objective function for the Optuna study
def objective(trial, args):

    # Instantiate a PChemicalTrainConfig with trial suggested hyperparameters
    config = BChemicalTrainConfig(
        hidden_size=trial.suggest_int("hidden_size", 50, 250),
        output_neurons=10 * trial.suggest_int("output_neurons", 10, 50),
        branch_hidden_layers=trial.suggest_int("branch_hidden_layers", 3, 7),
        trunk_hidden_layers=trial.suggest_int("trunk_hidden_layers", 3, 7),
        architecture=trial.suggest_categorical(
            "architecture", ["both", "branch", "trunk"]
        ),
        learning_rate=3e-5,
        optuna_trial=trial,
        device=args.device,
        num_epochs=100,
    )

    # Load data and create dataloaders
    train_data = np.load("data/branca_data/train_data_1e5.npy")
    test_data = np.load("data/branca_data/test_data_3e4.npy")
    timesteps = np.linspace(0, 15, 16)

    dataloader_train = create_dataloader_chemicals(
        train_data, timesteps, fraction=1, batch_size=config.batch_size, shuffle=True
    )

    dataloader_test = create_dataloader_chemicals(
        test_data, timesteps, fraction=1, batch_size=config.batch_size, shuffle=False
    )

    # Train the model
    _, _, test_loss = train_multionet_chemical(
        config, dataloader_train, dataloader_test
    )

    return np.mean(test_loss[-5:])


# Optuna study setup
def run(args):
    study_name = "multionet_bchemicals_fixedlr"
    storage_name = f"sqlite:///optuna/{study_name}.db"
    SEED = 42
    STUDY = False

    if STUDY:

        pruner = optuna.pruners.MedianPruner(n_warmup_steps=10, n_startup_trials=10)
        sampler = optuna.samplers.TPESampler(seed=SEED)

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
        )
        objective_with_args = partial(objective, args=args)
        study.optimize(objective_with_args, n_trials=200)

    else:
        study = optuna.load_study(study_name=study_name, storage=storage_name)

    print("Best trial:")
    print("  Value: ", study.best_trial.value)
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    # Visualization (Optional)
    plot_optimization_history(study).show()
    plot_param_importances(study).show()
    plot_parallel_coordinate(study).show()
    plot_intermediate_values(study).show()


if __name__ == "__main__":
    run(None)
