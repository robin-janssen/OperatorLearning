# This script implements an OPTUNA study to optimize the hyperparameters of a MultiONet for the priestley dataset.


# Import necessary libraries
import optuna
import os
import numpy as np
from functools import partial

from data import create_dataloader_chemicals
from training import PChemicalTrainConfig, train_multionet_chemical
from optuna.visualization import plot_optimization_history, plot_param_importances
from multionet_scripts.multionet_pchemicals_tests import prepare_priestley_data


# Define the objective function for the Optuna study
def objective(trial, args):

    # Instantiate a PChemicalTrainConfig with trial suggested hyperparameters
    config = PChemicalTrainConfig(
        hidden_size=trial.suggest_int("hidden_size", 300, 3000),
        output_neurons=216 * trial.suggest_int("output_neurons", 5, 20),
        branch_hidden_layers=trial.suggest_int("branch_hidden_layers", 3, 7),
        trunk_hidden_layers=trial.suggest_int("trunk_hidden_layers", 3, 7),
        learning_rate=trial.suggest_float("learning_rate", 1e-7, 1e-3, log=True),
        num_epochs=100,
        batch_size=256,
        optuna_trial=trial,
        device=args.device,
    )

    # Load data and create dataloaders
    data_folder = "data/chemicals_priestley"
    train_data = np.load(os.path.join(data_folder, "chemicals_train.npy"))
    test_data = np.load(os.path.join(data_folder, "chemicals_test.npy"))

    train_data, test_data, timesteps = prepare_priestley_data(
        train_data, test_data, train_cut=1000
    )

    dataloader_train = create_dataloader_chemicals(
        train_data, timesteps, fraction=1, batch_size=config.batch_size, shuffle=True
    )
    # dataloader_test = create_dataloader_chemicals(
    #     test_data, timesteps, fraction=1, batch_size=config.batch_size, shuffle=False
    # )

    # Train the model
    _, train_loss, _ = train_multionet_chemical(
        config, dataloader_train  # , dataloader_test
    )

    return np.mean(
        # test_loss[-10:]
        train_loss[-5:]
    )  # Use the average of the last 10 epochs' test loss as the objective value


# Optuna study setup
def run(args):
    study_name = "multionet_pchemicals_2"
    storage_name = f"sqlite:///optuna/{study_name}.db"
    SEED = 42

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5, n_startup_trials=10)
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
    study.optimize(objective_with_args, n_trials=100)

    print("Best trial:")
    print("  Value: ", study.best_trial.value)
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    # Visualization (Optional)
    plot_optimization_history(study).show()
    plot_param_importances(study).show()


if __name__ == "__main__":
    run(None)
