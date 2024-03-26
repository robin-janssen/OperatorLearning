# This script is used to train a MultiONet for the chemical dataset and visualize the results.

import numpy as np

# from torchinfo import summary

from data import chemicals, create_dataloader_chemicals, load_chemical_data
from plotting import (
    plot_chemical_examples,
    plot_chemicals_comparative,
    plot_losses,
    plot_chemical_results,
    plot_chemical_errors,
    plot_relative_errors_over_time,
)
from training import (
    train_multionet_chemical,
    test_deeponet,
    load_multionet,
    save_model,
)
from utils import read_yaml_config


def run(args):
    TRAIN = args.train
    VIS = args.vis
    USE_MASS_CONSERVATION = True
    pretrained_model_path = None  # "models/03-02/multionet_chemical_500_400e.pth"
    branch_input_size = 29
    trunk_input_size = 1
    hidden_size = 100
    branch_hidden_layers = 5
    trunk_hidden_layers = 5
    num_epochs = 200
    learning_rate = 3e-4
    fraction = 1
    output_neurons = 290  # number of neurons in the last layer of MODeepONet
    N_outputs = 29  # number of outputs of MODeepONet
    architecture = "both"  # "both", "branch", "trunk"
    device = args.device
    regularization_factor = 0.013
    massloss_factor = 0.013

    if USE_MASS_CONSERVATION:
        from data import masses
    else:
        masses = None

    # data = load_chemical_data("data/dataset100")
    data = load_chemical_data("data/dataset1000")
    data_shape = data.shape
    print(f"Data shape: {data_shape}")

    # Use only the amount of each chemical, not the gradients
    data = data[:, :, :29]

    N_timesteps = data.shape[1]

    # Split the data into training and testing (80/20)
    # train_data = data[: int(0.8 * data.shape[0])]
    # test_data = data[int(0.8 * data.shape[0]) :]
    train_data = data[:500]
    test_data = data[500:550]

    extracted_chemicals = chemicals.split(", ")

    if VIS:
        plot_chemical_examples(data, extracted_chemicals)
        plot_chemicals_comparative(data, extracted_chemicals)
    timesteps = np.arange(data.shape[1])

    dataloader_train = create_dataloader_chemicals(
        train_data, timesteps, fraction=1, batch_size=32, shuffle=True
    )

    dataloader_test = create_dataloader_chemicals(
        test_data, timesteps, fraction=1, batch_size=32, shuffle=False
    )

    if TRAIN:
        multionet, train_loss, test_loss = train_multionet_chemical(
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
            test_loader=dataloader_test,
            N_sensors=branch_input_size,
            N_timesteps=N_timesteps,
            schedule=False,
            architecture=architecture,
            pretrained_model_path=pretrained_model_path,
            device=device,
            visualize=False,
            regularization_factor=regularization_factor,
            massloss_factor=massloss_factor,
        )

        # Make sure that the loss history and train time are correct in case of pretrained model
        if pretrained_model_path is not None:
            config = read_yaml_config(pretrained_model_path)
            train_time = train_multionet_chemical.duration
            train_time += config["train_duration"]
            prev_train_loss, prev_test_loss = np.load(
                pretrained_model_path.replace(".pth", "_losses.npz")
            ).values()
            train_loss = np.concatenate((prev_train_loss, train_loss))
            test_loss = np.concatenate((prev_test_loss, test_loss))
            num_epochs += config["num_epochs"]
        else:
            train_time = train_multionet_chemical.duration

        # Save the MulitONet
        save_model(
            multionet,
            "multionet_chemical_optuna_200e",
            {
                "branch_input_size": branch_input_size,
                "trunk_input_size": trunk_input_size,
                "hidden_size": hidden_size,
                "branch_hidden_layers": branch_hidden_layers,
                "trunk_hidden_layers": trunk_hidden_layers,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "N_timesteps": N_timesteps,
                "fraction": fraction,
                "num_samples_train": train_data.shape[0],
                "num_samples_test": test_data.shape[0],
                "train_duration": train_time,
                "architecture": architecture,
            },
            train_loss=train_loss,
            test_loss=test_loss,
        )

        plot_losses(
            (train_loss, test_loss),
            ("Train loss", "Test loss"),
            "Losses (MultiONet for Chemicals)",
        )

    else:
        multionet = load_multionet(
            "models/03-08/multionet_chemical_fine_200e.pth",
            branch_input_size,
            trunk_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_hidden_layers,
            output_neurons,
            N_outputs,
            architecture="both",
        )

    # Print the model summary
    # summary(multionet_coeff, input_size=[(32, 29), (32, 1)], depth=1)

    # Predict the test data and calculate the errors
    average_error, predictions, ground_truth = test_deeponet(multionet, dataloader_test)

    print(f"Average error: {average_error}")

    predictions = predictions.reshape(-1, N_timesteps, N_outputs)
    ground_truth = ground_truth.reshape(-1, N_timesteps, N_outputs)
    errors = np.abs(predictions - ground_truth)
    relative_errors = errors / np.abs(ground_truth)

    # Plot the relative errors over time
    plot_relative_errors_over_time(
        relative_errors, "Relative errors over time (MultiONet for Chemicals)"
    )

    errors = np.mean(errors, axis=0)
    relative_errors = np.mean(relative_errors, axis=0)

    # Plot the results
    plot_chemical_results(
        predictions,
        ground_truth,
        extracted_chemicals,
        "MultiONet for Chemicals",
        num_chemicals=10,
    )

    # Plot the errors
    plot_chemical_errors(errors, extracted_chemicals, num_chemicals=10)
    plot_chemical_errors(relative_errors, extracted_chemicals, num_chemicals=10)

    print("Done!")
