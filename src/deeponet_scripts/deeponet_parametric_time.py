# This script investigates the performance of DeepONet on time-dependent data, where the time-dependence differs for each datapoint.
# The DeepONet receives an extra input (the decay rate).

import numpy as np

# from data import generate_decaying_sines, surface_plot
from data import generate_random_decaying_sines
from plotting import heatmap_plot, plot_functions_only
from training import (
    train_deeponet_visualized,
    load_deeponet,
    test_deeponet,
)

from plotting import plot_losses

from data import create_dataloader_2D_frac
from training import save_model


if __name__ == "__main__":
    TRAIN = False
    branch_input_size = 32
    trunk_input_size = 2
    hidden_size = 40
    branch_hidden_layers = 3
    trunk_hidden_layers = 3
    num_epochs = 400
    learning_rate = 3e-4
    N_timesteps = 16
    decay_rate = 1
    fraction = 0.25
    num_samples_train = 800
    num_samples_test = 200

    sensor_locations = np.linspace(0, 1, branch_input_size - 1)
    # surface_plot(sensor_locations, timesteps, polynomials, num_samples_to_plot=3)
    # train_data, amplitudes, frequencies, timesteps = generate_decaying_sines(
    #     sensor_locations=sensor_locations,
    #     decay_rate=decay_rate,
    #     num_samples=num_samples_train,
    #     N_steps=N_timesteps)
    # test_data, _, _, _ = generate_decaying_sines(
    #     sensor_locations=sensor_locations,
    #     decay_rate=decay_rate,
    #     num_samples=num_samples_test,
    #     N_steps=N_timesteps)
    if TRAIN:
        train_data, amplitudes, frequencies, timesteps = generate_random_decaying_sines(
            sensor_locations=sensor_locations,
            num_samples=num_samples_train,
            N_steps=N_timesteps,
        )

        dataloader_param = create_dataloader_2D_frac(
            train_data,
            sensor_locations,
            timesteps,
            batch_size=32,
            shuffle=True,
            fraction=fraction,
        )

        dataloader_no_param = create_dataloader_2D_frac(
            train_data[:, :-1, :],
            sensor_locations,
            timesteps,
            batch_size=32,
            shuffle=True,
            fraction=fraction,
        )

    test_data, _, _, timesteps = generate_random_decaying_sines(
        sensor_locations=sensor_locations,
        num_samples=num_samples_test,
        N_steps=N_timesteps,
    )

    dataloader_test_param = create_dataloader_2D_frac(
        test_data, sensor_locations, timesteps, batch_size=32, shuffle=False
    )

    dataloader_test_no_param = create_dataloader_2D_frac(
        test_data[:, :-1, :],
        sensor_locations,
        timesteps,
        batch_size=32,
        shuffle=False,
    )
    print("DataLoader created.")

    plot_functions_only(test_data[:, :-1, 0], sensor_locations, num_samples_test)

    # surface_plot(sensor_locations, timesteps, sines, num_samples_to_plot=3, title="Decaying sines")

    heatmap_plot(
        sensor_locations,
        timesteps,
        test_data[:, :-1, :],
        num_samples_to_plot=4,
        title="Decaying sines",
    )

    # Create a DataLoader for DeepONet

    # Now we need to train/load the DeepONet
    if TRAIN:
        # Train the DeepONet
        param_deeponet, loss, test_loss = train_deeponet_visualized(
            dataloader_param,
            branch_input_size,
            trunk_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_hidden_layers,
            num_epochs,
            learning_rate,
            test_loader=dataloader_test_param,
        )
        no_param_deeponet, np_loss, np_test_loss = train_deeponet_visualized(
            dataloader_no_param,
            branch_input_size - 1,
            trunk_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_hidden_layers,
            num_epochs,
            learning_rate,
            test_loader=dataloader_test_no_param,
        )
        # Plot the loss
        plot_losses(
            (loss, np.repeat(test_loss, 10), np_loss, np.repeat(np_test_loss, 10)),
            ("train loss", "test loss", "train loss no param", "test loss no param"),
        )
        # Save the trained DeepONets
        save_model(
            param_deeponet,
            "deeponet_params",
            {
                "branch_input_size": branch_input_size,
                "trunk_input_size": trunk_input_size,
                "hidden_size": hidden_size,
                "branch_hidden_layers": branch_hidden_layers,
                "trunk_hidden_layers": trunk_hidden_layers,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "N_timesteps": N_timesteps,
                "decay_rate": decay_rate,
                "fraction": fraction,
                "num_samples_train": num_samples_train,
                "num_samples_test": num_samples_test,
            },
        )
        save_model(
            no_param_deeponet,
            "deeponet_no_params",
            {
                "branch_input_size": branch_input_size - 1,
                "trunk_input_size": trunk_input_size,
                "hidden_size": hidden_size,
                "branch_hidden_layers": branch_hidden_layers,
                "trunk_hidden_layers": trunk_hidden_layers,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "N_timesteps": N_timesteps,
                "decay_rate": decay_rate,
                "fraction": fraction,
                "num_samples_train": num_samples_train,
                "num_samples_test": num_samples_test,
            },
        )

    else:
        # Load the DeepONet
        param_deeponet = load_deeponet(
            "models/04-02/deeponet_params.pth",
            branch_input_size,
            trunk_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_hidden_layers,
        )
        no_param_deeponet = load_deeponet(
            "models/04-02/deeponet_no_params.pth",
            branch_input_size - 1,
            trunk_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_hidden_layers,
        )

    # Test the DeepONet
    param_loss, param_predictions = test_deeponet(param_deeponet, dataloader_test_param)
    print(f"Total loss with parameter: {param_loss:.3E}")
    no_param_loss, no_param_predictions = test_deeponet(
        no_param_deeponet, dataloader_test_no_param
    )
    print(f"Total loss without parameter: {no_param_loss:.3E}")

    # Plot the results
    param_predictions = param_predictions.reshape(
        -1, len(sensor_locations), len(timesteps)
    )
    no_param_predictions = no_param_predictions.reshape(
        -1, len(sensor_locations), len(timesteps)
    )

    # surface_plot(sensor_locations, timesteps, sines, 3, predictions, title="DeepONet results")
    heatmap_plot(
        sensor_locations,
        timesteps,
        test_data[:, :-1, :],
        5,
        param_predictions,
        title="DeepONet with parameter",
    )

    heatmap_plot(
        sensor_locations,
        timesteps,
        test_data[:, :-1, :],
        5,
        no_param_predictions,
        title="DeepONet without parameter",
    )

    print("Done.")
