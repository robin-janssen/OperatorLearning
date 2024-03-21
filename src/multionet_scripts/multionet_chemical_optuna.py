# This script is used to optimize the hyperparameters of the multionet_chemical model using optuna.
# The results of the optimization are stored in optuna/multionet_chemical_fine.db

import optuna
from optuna.visualization import (
    plot_param_importances,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_intermediate_values,
)
import numpy as np
import pickle

from training import (
    create_dataloader_chemicals,
    train_multionet_chemical,
    test_deeponet,
)
from utils import load_chemical_data

import plotly.io as pio

pio.renderers.default = "browser"


def objective_1(trial):
    """
    This is the objective function for the first (rough) optimization run.
    It is used to find some good hyperparameters to start with.
    The results of this run are stored in optuna/multionet_chemical.db
    """
    # Define the constants
    branch_input_size = 29
    trunk_input_size = 1
    N_outputs = 29
    num_epochs = 30
    masses = None
    device = "cpu"
    pretrained_model_path = None

    # Define the hyperparameter space
    branch_hidden_layers = trial.suggest_int("branch_hidden_layers", 1, 5)
    trunk_hidden_layers = trial.suggest_int("trunk_hidden_layers", 1, 5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    hidden_size = trial.suggest_int("hidden_size", 20, 200)
    output_neurons_factor = trial.suggest_int("output_neurons_factor", 1, 20)
    architecture = trial.suggest_categorical(
        "architecture", ["both", "branch", "trunk"]
    )

    output_neurons = output_neurons_factor * branch_input_size

    # Load the data
    data = load_chemical_data("data/dataset1000")
    data = data[:, :, :29]
    train_data = data[200:250]
    test_data = data[100:150]

    timesteps = np.arange(data.shape[1])
    N_timesteps = data.shape[1]

    # Create the dataloaders
    dataloader_train = create_dataloader_chemicals(
        train_data, timesteps, fraction=1, batch_size=32, shuffle=True
    )

    dataloader_test = create_dataloader_chemicals(
        test_data, timesteps, fraction=1, batch_size=32, shuffle=False
    )

    multionet, _ = train_multionet_chemical(
        dataloader_train,
        masses,
        branch_input_size,
        trunk_input_size,
        hidden_size,
        branch_hidden_layers,
        trunk_hidden_layers,
        output_neurons,
        N_outputs,
        num_epochs,
        learning_rate,
        test_loader=None,
        N_sensors=branch_input_size,
        N_timesteps=N_timesteps,
        schedule=False,
        architecture=architecture,
        pretrained_model_path=pretrained_model_path,
        device=device,
        visualize=False,
        optuna_trial=trial,
    )

    loss, _, _ = test_deeponet(multionet, dataloader_test)

    return loss


def objective_2(trial):
    """
    This is the objective function for the second (refinement) run.
    It is used to further optimize the hyperparameters found in the first run.
    The results of this run are stored in optuna/multionet_chemical_fine.db
    """

    # Define the constants
    branch_input_size = 29
    trunk_input_size = 1
    N_outputs = 29
    num_epochs = 50
    hidden_size = 100
    masses = None
    device = "cpu"
    pretrained_model_path = None
    output_neurons = 290
    architecture = "both"

    # Define the hyperparameter space
    branch_hidden_layers = trial.suggest_int("branch_hidden_layers", 3, 5)
    trunk_hidden_layers = trial.suggest_int("trunk_hidden_layers", 3, 5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    regularization_factor = trial.suggest_float(
        "regularization_factor", 1e-4, 1e-1, log=True
    )
    massloss_factor = trial.suggest_float("massloss_factor", 1e-4, 1e-1, log=True)

    # Load the data
    data = load_chemical_data("data/dataset1000")
    data = data[:, :, :29]
    train_data = data[200:250]

    timesteps = np.arange(data.shape[1])
    N_timesteps = data.shape[1]

    # Create the dataloaders
    dataloader_train = create_dataloader_chemicals(
        train_data, timesteps, fraction=1, batch_size=32, shuffle=True
    )

    _, train_loss_history = train_multionet_chemical(
        dataloader_train,
        masses,
        branch_input_size,
        trunk_input_size,
        hidden_size,
        branch_hidden_layers,
        trunk_hidden_layers,
        output_neurons,
        N_outputs,
        num_epochs,
        learning_rate,
        test_loader=None,
        N_sensors=branch_input_size,
        N_timesteps=N_timesteps,
        schedule=False,
        architecture=architecture,
        pretrained_model_path=pretrained_model_path,
        device=device,
        visualize=False,
        optuna_trial=trial,
        regularization_factor=regularization_factor,
        massloss_factor=massloss_factor,
    )

    loss = train_loss_history[-1]

    return loss


if __name__ == "__main__":

    SEED = 42
    np.random.seed(SEED)
    resumeStudy = False
    VIS = True
    study_name = "multionet_chemical_fine"
    objective = objective_2

    if not VIS:

        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5, n_startup_trials=10)

        if resumeStudy:
            sampler = pickle.load(open("optuna/sampler.pkl", "rb"))
            study = optuna.create_study(
                direction="minimize",
                study_name=study_name,
                sampler=sampler,
                pruner=pruner,
                storage=f"sqlite:///optuna/{study_name}.db",
                load_if_exists=True,
            )
        else:
            sampler = optuna.samplers.TPESampler(seed=SEED)
            with open("optuna/sampler.pkl", "wb") as f:
                pickle.dump(sampler, f)
            study = optuna.create_study(
                direction="minimize",
                study_name=study_name,
                sampler=sampler,
                pruner=pruner,
                storage=f"sqlite:///optuna/{study_name}.db",
            )
        study.optimize(objective, n_trials=100)  # Adjust the number of trials as needed

        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

    else:
        study = optuna.load_study(
            study_name=study_name,
            storage=f"sqlite:///optuna/{study_name}.db",
        )
        print(study.best_params)
        plot = plot_optimization_history(study)
        plot.show()
        # plot2 = plot_contour(study)
        # plot2.show()
        plot3 = plot_param_importances(study)
        plot3.show()
        plot4 = plot_parallel_coordinate(study)
        plot4.show()
        plot5 = plot_intermediate_values(study)
        plot5.show()
