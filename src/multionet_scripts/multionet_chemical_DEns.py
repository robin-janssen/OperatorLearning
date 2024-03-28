import os
import yaml
import numpy as np

from training import load_multionet, test_deeponet
from plotting import (
    plot_losses,
    visualise_deep_ensemble,
)
from data import create_dataloader_chemicals, load_chemical_data
from data.osu_chemicals import chemicals


def load_model_and_losses(directory):
    model_files = [f for f in os.listdir(directory) if f.endswith(".pth")]

    models = []
    configs = []
    losses = []

    for model_file in model_files:
        base_name = model_file[:-4]  # Remove the .pth extension to get the base name

        # Load model
        config_path = os.path.join(directory, base_name + ".yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)  # Load model configuration
        configs.append(config)

        # Initialize the model based on the configuration
        model_path = os.path.join(directory, base_name + ".pth")
        model = load_multionet(
            model_path,
            config["branch_input_size"],
            config["trunk_input_size"],
            config["hidden_size"],
            config["branch_hidden_layers"],
            config["trunk_hidden_layers"],
            config["output_neurons"],
            config["N_outputs"],
            config["architecture"],
        )
        models.append(model)

        # Load losses
        losses_path = os.path.join(directory, base_name + "_losses.npz")
        losses_data = np.load(losses_path)
        train_loss = losses_data["train_loss"]
        test_loss = losses_data["test_loss"]
        losses.append({"train_loss": train_loss, "test_loss": test_loss})

    return models, configs, losses


def deep_ensemble_uq(models, configs, dataloader):

    preds_list = []
    errors_list = []
    relative_errors_list = []

    for idx, (model, config) in enumerate(zip(models, configs)):
        total_loss, preds, targets = test_deeponet(model, dataloader)
        print(f"Average prediction error (DeepONet {idx}): {total_loss:.3E}")
        N_outputs = config["N_outputs"]
        N_timesteps = config["N_timesteps"]
        preds = preds.reshape(-1, N_timesteps, N_outputs)
        targets = targets.reshape(-1, N_timesteps, N_outputs)
        errors = np.abs(preds - targets)
        relative_errors = errors / np.abs(targets)

        preds_list.append(preds)
        errors_list.append(errors)
        relative_errors_list.append(relative_errors)

    return preds_list, targets, errors_list, relative_errors_list


def deep_ensemble_losses(losses):
    losses_list = []
    names_list = []
    for idx, loss in enumerate(losses):
        losses_list.append(loss["train_loss"])
        names_list.append(f"train_loss_{idx}")
        losses_list.append(loss["test_loss"])
        names_list.append(f"test_loss_{idx}")
    return losses_list, names_list


# Example usage


def run(args):
    directory = "models/03-32/"
    directory = os.path.join(os.getcwd(), directory)
    models, configs, losses = load_model_and_losses(directory)

    data = load_chemical_data(args.data_path)
    data = data[:, :, :29]
    test_data = data[500:550]
    timesteps = np.arange(data.shape[1])

    dataloader_test = create_dataloader_chemicals(
        test_data, timesteps, fraction=1, batch_size=32, shuffle=False
    )

    losses_val_list, losses_label_list = deep_ensemble_losses(losses)

    plot_losses(
        losses_val_list, losses_label_list, "Losses (MultiONet Ensemble for Chemicals)"
    )

    preds_list, targets, errors_list, relative_errors_list = deep_ensemble_uq(
        models, configs, dataloader_test
    )

    visualise_deep_ensemble(
        preds_list,
        targets,
        num_chemicals=5,
        chemical_names=chemicals,
    )

    print("Done!")


if __name__ == "__main__":
    run()
