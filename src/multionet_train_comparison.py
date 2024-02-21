# This script compares two training approaches for MultiONet.
# The first approach calculates the loss based on the coefficients of the polynomial functions.
# The second approach calculates the loss based on the polynomial functions themselves. To do so, the coefficients are evaluated at the sensor locations and the loss is calculated based on the resulting values.

import numpy as np

# from datagen import generate_decaying_sines, surface_plot
from datagen import generate_decaying_polynomials
from plotting import (
    heatmap_plot,
    heatmap_plot_errors,
    plot_functions_only,
    surface_plot,
    plot_losses,
)
from training import (
    train_multionet_visualized,
    train_multionet_poly_visualized,
    load_multionet,
    test_multionet_poly,
)
from utils import save_model
from training import create_dataloader_2D_frac_coeff

if __name__ == "__main__":
    TRAIN = False
    VIS = False
    branch_input_size = 11
    N_timesteps = 11
    trunk_input_size = 1
    hidden_size = 40
    branch_hidden_layers = 3
    trunk_hidden_layers = 3
    num_epochs = 200
    learning_rate = 3e-4
    decay_rate = 1
    fraction = 1
    num_samples_train = 5000
    num_samples_test = 500
    output_neurons = 60  # number of neurons in the last layer of MODeepONet
    N_outputs = 6  # number of outputs of MODeepONet

    sensor_locations = np.linspace(0, 1, branch_input_size)

    train_data, train_coeffs, train_timesteps = generate_decaying_polynomials(
        sensor_locations=sensor_locations,
        decay_rate=decay_rate,
        num_samples=num_samples_train,
        N_steps=N_timesteps,
    )
    test_data, test_coeffs, test_timesteps = generate_decaying_polynomials(
        sensor_locations=sensor_locations,
        decay_rate=decay_rate,
        num_samples=num_samples_test,
        N_steps=N_timesteps,
    )
    print("Data generated.")

    if VIS:
        surface_plot(
            sensor_locations,
            timesteps=train_timesteps,
            functions=train_data,
            num_samples_to_plot=3,
            title="Decaying polynomial functions",
        )
        heatmap_plot(
            sensor_locations,
            timesteps=train_timesteps,
            functions=train_data,
            num_samples_to_plot=3,
            title="Decaying polynomial functions",
        )
        plot_functions_only(train_data, sensor_locations, 100)

    # Create the DataLoaders
    dataloader = create_dataloader_2D_frac_coeff(
        train_data,
        train_coeffs,
        sensor_locations,
        train_timesteps,
        batch_size=32,
        shuffle=True,
        fraction=fraction,
    )

    dataloader_test = create_dataloader_2D_frac_coeff(
        test_data,
        test_coeffs,
        sensor_locations,
        test_timesteps,
        batch_size=32,
        shuffle=False,
        fraction=1,
    )

    print("DataLoader created.")

    # Now we need to train/load the DeepONet
    if TRAIN:
        # Train a MODeepONet where both the branch and trunk networks are split
        multionet_coeff, loss_coeff, test_loss_coeff = train_multionet_visualized(
            dataloader,
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
            architecture="both",
        )

        # Save the MODeepONet (trained on coefficients)
        save_model(
            multionet_coeff,
            "multionet_both",
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
                "train_duration": train_multionet_visualized.duration,
                "architecture": "both",
            },
        )

        # Train a MODeepONet where only the branch network is split
        multionet_poly, loss_poly, test_loss_poly = train_multionet_poly_visualized(
            dataloader,
            branch_input_size,
            trunk_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_hidden_layers,
            output_neurons,
            N_outputs,
            num_epochs,
            learning_rate,
            schedule=False,
            test_loader=dataloader_test,
            sensor_locations=sensor_locations,
            N_timesteps=N_timesteps,
            architecture="both",
        )

        # Save the MODeepONet (trained on polynomials)
        save_model(
            multionet_poly,
            "multionet_poly",
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
                "fraction": 1,
                "num_samples_train": num_samples_train,
                "num_samples_test": num_samples_test,
                "train_duration": train_multionet_visualized.duration,
                "architecture": "both",
            },
        )

        # Plot the losses (and save the plot)
        plot_losses(
            (loss_coeff, test_loss_coeff, loss_poly, test_loss_poly),
            ("Train (coeff)", "Test (coeff)", "Train (poly)", "Test (poly)"),
        )

    else:
        # Load the DeepONet
        multionet_coeff = load_multionet(
            "models/21-02/multionet_coeff.pth",
            branch_input_size,
            trunk_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_hidden_layers,
            output_neurons,
            N_outputs,
            architecture="both",
        )

        multionet_poly = load_multionet(
            "models/21-02/multionet_poly.pth",
            branch_input_size,
            trunk_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_hidden_layers,
            output_neurons,
            N_outputs,
            architecture="both",
        )

    # Test the different MODeepONets and plot some results

    # First the MODeepONet with both the branch and trunk networks split
    total_loss, preds_coeff, targets_coeff = test_multionet_poly(
        multionet_coeff, dataloader_test, sensor_locations
    )
    print(f"Average prediction error (MultiONet coeff): {total_loss:.3E}")

    preds_coeff = preds_coeff.reshape(
        -1, len(test_timesteps), len(sensor_locations)
    ).transpose(0, 2, 1)
    targets_coeff = targets_coeff.reshape(
        -1, len(test_timesteps), len(sensor_locations)
    ).transpose(0, 2, 1)

    heatmap_plot(
        sensor_locations,
        test_timesteps,
        targets_coeff,
        5,
        preds_coeff,
        title="MultiONet (both) results",
    )

    # Now the MODeepONet with only the branch network split

    total_loss, preds_poly, targets_poly = test_multionet_poly(
        multionet_poly, dataloader_test, sensor_locations
    )

    print(f"Average prediction error (Multionet poly): {total_loss:.3E}")

    preds_poly = preds_poly.reshape(
        -1, len(test_timesteps), len(sensor_locations)
    ).transpose(0, 2, 1)
    targets_poly = targets_poly.reshape(
        -1, len(test_timesteps), len(sensor_locations)
    ).transpose(0, 2, 1)

    heatmap_plot(
        sensor_locations,
        test_timesteps,
        targets_poly,
        5,
        preds_poly,
        title="MultiONet (branch) results",
    )

    # Compare the model errors

    errors = []
    errors.append(np.abs(preds_coeff - targets_coeff))
    errors.append(np.abs(preds_poly - targets_poly))

    heatmap_plot_errors(
        sensor_locations,
        test_timesteps,
        errors,
        5,
        title="MultiONet errors",
    )

    print("Done.")
