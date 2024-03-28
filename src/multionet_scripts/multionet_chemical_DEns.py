from training import load_multionet
from plotting import plot_losses

import os
import yaml
import numpy as np  # Make sure to import your model class


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


# Example usage


def run(args):
    directory = "src/models/03-26/"
    directory = os.path.join(os.path.dirname(os.getcwd()), directory)
    models, configs, losses = load_model_and_losses(directory)

    losses_list = []
    names_list = []
    error_list = []

    for idx, (model, config, loss) in enumerate(zip(models, configs, losses)):
        losses_list.append(loss["train_loss"])
        names_list.append(f"train_loss_{idx}")
        losses_list.append(loss["test_loss"])
        names_list.append(f"test_loss_{idx}")

    plot_losses(losses_list, names_list)
