import numpy as np
import torch

from datagen import generate_GRF_data, generate_polynomial_data, generate_sine_data

from training import (
    train_deeponet,
    plot_losses,
    load_deeponet,
    test_deeponet,
    create_dataloader,
)

from plotting import plot_results


if __name__ == "__main__":

    # Hyperparameters
    TRAIN = True
    branch_input_size = 21
    trunk_input_size = 1
    hidden_size = 40
    branch_output_size = hidden_size
    trunk_output_size = hidden_size
    branch_hidden_layers = 3
    trunk_hidden_layers = 1
    dataset_size = 1000
    num_epochs = 10
    sensor_points = np.linspace(0, 1, branch_input_size)
    num_samples_to_plot = 3

    if TRAIN:
        # Generate polynomial and GRF data and create DataLoaders
        poly_data = generate_polynomial_data(dataset_size, sensor_points, scale=3)
        grf_data = generate_GRF_data(dataset_size, sensor_points, length_scale=0.3)
        poly_data_loader = create_dataloader(poly_data, sensor_points, shuffle=True)
        grf_data_loader = create_dataloader(grf_data, sensor_points, shuffle=True)

        # Train models
        poly_deeponet, poly_loss = train_deeponet(
            poly_data_loader,
            branch_input_size,
            trunk_input_size,
            hidden_size,
            num_epochs,
            branch_hidden_layers,
            trunk_hidden_layers,
        )
        grf_deeponet, grf_loss = train_deeponet(
            grf_data_loader,
            branch_input_size,
            trunk_input_size,
            hidden_size,
            num_epochs,
            branch_hidden_layers,
            trunk_hidden_layers,
        )
        torch.save(poly_deeponet.state_dict(), "poly_deeponet.pt")
        torch.save(grf_deeponet.state_dict(), "grf_deeponet.pt")

        # Plot the loss trajectories
        plot_losses([poly_loss, grf_loss], ["Polynomial Data", "GRF Data"])
    else:
        poly_deeponet = load_deeponet(
            "poly_deeponet.pt",
            branch_input_size,
            trunk_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_hidden_layers,
        )
        grf_deeponet = load_deeponet(
            "grf_deeponet.pt",
            branch_input_size,
            trunk_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_hidden_layers,
        )

    # Generate sine data for testing
    sine_data = generate_sine_data(1000, sensor_points)
    sine_data_loader = create_dataloader(sine_data, sensor_points)

    # Generate some more polynomial data and GRF data for testing
    poly_test_data = generate_polynomial_data(dataset_size, sensor_points, scale=3)
    grf_test_data = generate_GRF_data(dataset_size, sensor_points, length_scale=0.3)
    poly_test_data_loader = create_dataloader(poly_test_data, sensor_points)
    grf_test_data_loader = create_dataloader(grf_test_data, sensor_points)

    # Test models
    poly_test_loss, poly_predictions = test_deeponet(poly_deeponet, sine_data_loader)
    grf_test_loss, grf_predictions = test_deeponet(grf_deeponet, sine_data_loader)
    print(f"Polynomial Model Test Loss: {poly_test_loss}")
    print(f"GRF Model Test Loss: {grf_test_loss}")

    # Plot some examples from the test set
    poly_predictions = poly_predictions.reshape(-1, len(sensor_points))
    grf_predictions = grf_predictions.reshape(-1, len(sensor_points))
    sine_data = np.array(sine_data)
    plot_results(
        "Polynomial data",
        sensor_points,
        sine_data[:, 0],
        sine_data[:, 1],
        sensor_points,
        poly_predictions,
        num_samples_to_plot,
    )
    plot_results(
        "GRF data",
        sensor_points,
        sine_data[:, 0],
        sine_data[:, 1],
        sensor_points,
        grf_predictions,
        num_samples_to_plot,
    )
