# This script is used to train and test a DeepONet on time-dependent data.

import numpy as np

# from datagen import generate_decaying_sines, surface_plot
from datagen import generate_oscillating_sines
from plotting import heatmap_plot, plot_functions_only, surface_plot, plot_losses
from training import (
    train_deeponet,
    load_deeponet,
    test_deeponet,
    create_dataloader_2D_frac,
)
from utils import save_model


if __name__ == "__main__":
    TRAIN = True
    branch_input_size = 11
    trunk_input_size = 2
    hidden_size = 40
    branch_hidden_layers = 3
    trunk_hidden_layers = 3
    num_epochs = 3
    learning_rate = 3e-4
    N_timesteps = 41
    decay_rate = 1
    fraction = 0.25
    num_samples_train = 800
    num_samples_test = 100

    sensor_locations = np.linspace(0, 1, branch_input_size)
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
    train_data, amplitudes, frequencies, timesteps = generate_oscillating_sines(
        sensor_locations=sensor_locations,
        rate=decay_rate,
        num_samples=num_samples_train,
        N_steps=N_timesteps,
    )
    test_data, _, _, _ = generate_oscillating_sines(
        sensor_locations=sensor_locations,
        rate=decay_rate,
        num_samples=num_samples_test,
        N_steps=N_timesteps,
    )
    surface_plot(
        sensor_locations,
        timesteps,
        train_data,
        num_samples_to_plot=3,
        title="Decaying sines",
    )
    heatmap_plot(
        sensor_locations,
        timesteps,
        train_data,
        num_samples_to_plot=3,
        title="Decaying sines",
    )
    print("Data generated.")
    plot_functions_only(train_data, sensor_locations, 100)

    # grid = subsampling_grid(len(sensor_locations), len(timesteps), fraction=fraction, visualize=True)

    # Create a DataLoader for DeepONet
    dataloader = create_dataloader_2D_frac(
        train_data,
        sensor_locations,
        timesteps,
        batch_size=32,
        shuffle=True,
        fraction=fraction,
    )

    dataloader_test = create_dataloader_2D_frac(
        test_data, sensor_locations, timesteps, batch_size=32, shuffle=False
    )
    print("DataLoader created.")

    # Now we need to train/load the DeepONet
    if TRAIN:
        # Train the DeepONet
        vanilla_deeponet, loss, test_loss = train_deeponet(
            dataloader,
            branch_input_size,
            trunk_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_hidden_layers,
            num_epochs,
            learning_rate,
            test_loader=dataloader_test,
            N_sensors=branch_input_size,
            N_timesteps=N_timesteps,
        )
        # Plot the loss
        plot_losses((loss, np.repeat(test_loss, 10)), ("train loss", "test loss"))
        # Save the trained DeepONet
        save_model(
            vanilla_deeponet,
            "deeponet_time_dependent",
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
    else:
        # Load the DeepONet
        vanilla_deeponet = load_deeponet(
            "models/31-01/deeponet_time_dependent.pth",
            branch_input_size,
            trunk_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_hidden_layers,
        )

    # Test the DeepONet
    total_loss, predictions, _ = test_deeponet(vanilla_deeponet, dataloader_test)
    print(f"Total loss: {total_loss:.3E}")

    # Plot the results
    predictions = predictions.reshape(-1, len(sensor_locations), len(timesteps))

    # surface_plot(sensor_locations, timesteps, sines, 3, predictions, title="DeepONet results")
    heatmap_plot(
        sensor_locations, timesteps, test_data, 5, predictions, title="DeepONet results"
    )

    print("Done.")
