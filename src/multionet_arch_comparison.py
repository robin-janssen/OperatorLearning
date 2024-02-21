# This script compares different architectures for the MODeepONet. The architectures are:
# - Both the branch and trunk networks are split
# - Only the branch network is split
# - Only the trunk network is split
# The script generates decaying polynomials and trains the different MODeepONets on the data. The trained MODeepONets are then tested and the results are plotted.

# Currently, it seems that splitting only the trunk network is the best option (best performance at equal training time).

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
    load_multionet,
    test_multionet_polynomial_old,
)
from utils import save_model
from training import create_dataloader_2D_frac_coeff

if __name__ == "__main__":
    TRAIN = False
    VIS = False
    branch_input_size = 31
    N_timesteps = 31
    trunk_input_size = 1
    hidden_size = 40
    branch_hidden_layers = 3
    trunk_hidden_layers = 3
    num_epochs = 100
    learning_rate = 3e-4
    decay_rate = 1
    fraction = 0.5
    num_samples_train = 10000
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
        shuffle=False,
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
        multionet_both, loss_both, test_loss_both = train_multionet_visualized(
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

        # Save the trained DeepONet
        save_model(
            multionet_both,
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
        multionet_branch, loss_branch, test_loss_branch = train_multionet_visualized(
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
            architecture="branch",
        )

        # Save the trained MODeepONet
        save_model(
            multionet_branch,
            "multionet_branch",
            {
                "branch_input_size": branch_input_size,
                "trunk_input_size": trunk_input_size,
                "hidden_size": hidden_size,
                "branch_hidden_layers": branch_hidden_layers,
                "trunk_hidden_layers": trunk_hidden_layers,
                "" "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "N_timesteps": N_timesteps,
                "decay_rate": decay_rate,
                "fraction": 1,
                "num_samples_train": num_samples_train,
                "num_samples_test": num_samples_test,
                "train_duration": train_multionet_visualized.duration,
                "architecture": "branch",
            },
        )

        multionet_trunk, loss_trunk, test_loss_trunk = train_multionet_visualized(
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
            architecture="trunk",
        )

        save_model(
            multionet_trunk,
            "multionet_trunk",
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
                "architecture": "trunk",
            },
        )

        # Plot the losses (and save the plot)
        plot_losses(
            (
                loss_both,
                test_loss_both,
                loss_branch,
                test_loss_branch,
                loss_trunk,
                test_loss_trunk,
            ),
            ("Both", "Both Test", "Branch", "Branch Test", "Trunk", "Trunk Test"),
        )

    else:
        # Load the DeepONet
        multionet_both = load_multionet(
            "models/21-02/multionet_both.pth",
            branch_input_size,
            trunk_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_hidden_layers,
            output_neurons,
            N_outputs,
            architecture="both",
        )

        multionet_branch = load_multionet(
            "models/21-02/multionet_branch.pth",
            branch_input_size,
            trunk_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_hidden_layers,
            output_neurons,
            N_outputs,
            architecture="branch",
        )

        multionet_trunk = load_multionet(
            "models/21-02/multionet_trunk.pth",
            branch_input_size,
            trunk_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_hidden_layers,
            output_neurons,
            N_outputs,
            architecture="trunk",
        )

    # Test the different MODeepONets and plot some results

    # First the MODeepONet with both the branch and trunk networks split
    total_loss, preds_both, targets_both = test_multionet_polynomial_old(
        multionet_both, dataloader_test, sensor_locations
    )
    print(f"Average prediction error (Multionet both): {total_loss:.3E}")

    preds_both = preds_both.reshape(
        -1, len(sensor_locations), len(test_timesteps)
    ).transpose(0, 2, 1)
    targets_both = targets_both.reshape(
        -1, len(sensor_locations), len(test_timesteps)
    ).transpose(0, 2, 1)

    heatmap_plot(
        sensor_locations,
        test_timesteps,
        targets_both,
        5,
        preds_both,
        title="MultiONet (both) results",
    )

    # Now the MODeepONet with only the branch network split

    total_loss, preds_branch, targets_branch = test_multionet_polynomial_old(
        multionet_branch, dataloader_test, sensor_locations
    )

    print(f"Average prediction error (Multionet branch): {total_loss:.3E}")

    preds_branch = preds_branch.reshape(
        -1, len(sensor_locations), len(test_timesteps)
    ).transpose(0, 2, 1)
    targets_branch = targets_branch.reshape(
        -1, len(sensor_locations), len(test_timesteps)
    ).transpose(0, 2, 1)

    heatmap_plot(
        sensor_locations,
        test_timesteps,
        targets_branch,
        5,
        preds_branch,
        title="MultiONet (branch) results",
    )

    # Finally the MODeepONet with only the trunk network split

    total_loss, preds_trunk, targets_trunk = test_multionet_polynomial_old(
        multionet_trunk, dataloader_test, sensor_locations
    )

    print(f"Average prediction error (Multionet trunk): {total_loss:.3E}")

    preds_trunk = preds_trunk.reshape(
        -1, len(sensor_locations), len(test_timesteps)
    ).transpose(0, 2, 1)
    targets_trunk = targets_trunk.reshape(
        -1, len(sensor_locations), len(test_timesteps)
    ).transpose(0, 2, 1)

    heatmap_plot(
        sensor_locations,
        test_timesteps,
        targets_trunk,
        5,
        preds_trunk,
        title="MultiONet (trunk) results",
    )

    # Compare the model errors

    errors = []
    errors.append(np.abs(preds_both - targets_both))
    errors.append(np.abs(preds_branch - targets_branch))
    errors.append(np.abs(preds_trunk - targets_trunk))

    heatmap_plot_errors(
        sensor_locations,
        test_timesteps,
        errors,
        5,
        title="MultiONet errors",
    )

    print("Done.")
