import os
import yaml
import numpy as np

from training import load_multionet, test_deeponet
from plotting import plot_losses
from data import create_dataloader_chemicals, load_chemical_data


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
    directory = "models/03-26/"
    directory = os.path.join(os.getcwd(), directory)
    models, configs, losses = load_model_and_losses(directory)

    data = load_chemical_data(args.data_path)
    data = data[:, :, :29]
    train_data, test_data = data[:500], data[500:550]
    timesteps = np.arange(data.shape[1])

    # dataloader_train = create_dataloader_chemicals(
    #     train_data, timesteps, fraction=1, batch_size=32, shuffle=True
    # )

    dataloader_test = create_dataloader_chemicals(
        test_data, timesteps, fraction=1, batch_size=32, shuffle=False
    )

    losses_list = []
    names_list = []
    total_loss_list = []
    preds_list = []
    targets_list = []

    for idx, (model, config, loss) in enumerate(zip(models, configs, losses)):
        losses_list.append(loss["train_loss"])
        names_list.append(f"train_loss_{idx}")
        losses_list.append(loss["test_loss"])
        names_list.append(f"test_loss_{idx}")
        total_loss, preds, targets = test_deeponet(model, dataloader_test)
        print(f"Average prediction error (DeepONet {idx}): {total_loss:.3E}")
        total_loss_list.append(total_loss)
        preds_list.append(preds)
        targets_list.append(targets)

    plot_losses(losses_list, names_list)

    print("Done!")
