# Import necessary libraries
import optuna
import os
import numpy as np

from data import create_dataloader_chemicals
from training import PChemicalTrainConfig, train_multionet_chemical
from utils import load_chemical_data
from optuna.visualization import plot_optimization_history, plot_param_importances
from multionet_scripts.multionet_chemical_tests import prepare_priestley_data


# Define the objective function for the Optuna study
def objective(trial):

    # Instantiate a PChemicalTrainConfig with trial suggested hyperparameters
    config = PChemicalTrainConfig(
        hidden_size=trial.suggest_int("hidden_size", 200, 2000),
        output_neurons=216 * trial.suggest_int("output_neurons", 5, 20),
        branch_hidden_layers=trial.suggest_int("branch_hidden_layers", 3, 7),
        trunk_hidden_layers=trial.suggest_int("trunk_hidden_layers", 3, 7),
        learning_rate=trial.suggest_float("learning_rate", 1e-7, 1e-3, log=True),
        num_epochs=100,
        batch_size=128,
        optuna_trial=trial,
    )

    # Load data and create dataloaders
    data_folder = "data/chemicals_priestley"
    train_data = np.load(os.path.join(data_folder, "chemicals_train.npy"))
    test_data = np.load(os.path.join(data_folder, "chemicals_test.npy"))

    train_data, test_data, timesteps = prepare_priestley_data(train_data, test_data)

    dataloader_train = create_dataloader_chemicals(
        train_data, timesteps, fraction=1, batch_size=config.batch_size, shuffle=True
    )
    dataloader_test = create_dataloader_chemicals(
        test_data, timesteps, fraction=1, batch_size=config.batch_size, shuffle=False
    )

    # Train the model
    multionet, train_loss, test_loss = train_multionet_chemical(
        config, dataloader_train, dataloader_test
    )

    return np.mean(
        test_loss[-5:]
    )  # Use the average of the last 10 epochs' test loss as the objective value


# Optuna study setup
def run():
    study_name = "multionet_pchemicals"
    storage_name = f"sqlite:///optuna/{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="minimize",
    )
    study.optimize(objective, n_trials=100)

    print("Best trial:")
    print("  Value: ", study.best_trial.value)
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    # Visualization (Optional)
    plot_optimization_history(study).show()
    plot_param_importances(study).show()


if __name__ == "__main__":
    run_study()
