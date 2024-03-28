# This script is a first test of a DeepONet with multiple outputs.
# Multiple outputs are obtained by splitting the last layer of the branch and trunk networks and performing the tensor product separately for each output.

import numpy as np

# import torch
# from torch.utils.data import DataLoader, TensorDataset
# import matplotlib.pyplot as plt

# from data import generate_decaying_sines, surface_plot
from data import (
    generate_decaying_polynomials,
    create_dataloader_2D,
    create_dataloader_2D_coeff,
)

from plotting import (
    heatmap_plot,
    plot_functions_only,
    surface_plot,
    plot_losses,
    plot_relative_errors_over_time,
)
from training import (
    train_deeponet_visualized,
    train_multionet_poly_coeff,
    load_deeponet,
    load_multionet,
    test_deeponet,
    test_multionet_poly,
    save_model,
)


def run(args):
    TRAIN = False
    branch_input_size = 81
    N_timesteps = 16
    trunk_input_size = 2
    hidden_size = 40
    branch_hidden_layers = 3
    trunk_hidden_layers = 3
    num_epochs = 100
    learning_rate = 3e-4
    decay_rate = 1
    fraction = 0.25  # 1
    num_samples_train = 1000  # 5000
    num_samples_test = 100  # 400
    output_neurons = 60  # number of neurons in the last layer of MODeepONet
    N_outputs = 6  # number of outputs of MODeepONet

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
    train_data, train_coeffs, train_timesteps = generate_decaying_polynomials(
        sensor_locations=sensor_locations,
        decay_rate=decay_rate,
        num_samples=num_samples_train,
        N_steps=N_timesteps,
    )
    train_data_mo, train_coeffs_mo, train_timesteps_mo = generate_decaying_polynomials(
        sensor_locations=sensor_locations,
        decay_rate=decay_rate,
        num_samples=num_samples_train * 5,
        N_steps=N_timesteps,
    )
    test_data, test_coeffs, test_timesteps = generate_decaying_polynomials(
        sensor_locations=sensor_locations,
        decay_rate=decay_rate,
        num_samples=num_samples_test,
        N_steps=N_timesteps,
    )
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
    print("Data generated.")
    plot_functions_only(train_data, sensor_locations, 100)

    # Create the DataLoaders
    dataloader_single = create_dataloader_2D(
        train_data,
        sensor_locations,
        train_timesteps,
        batch_size=32,
        shuffle=True,
        fraction=fraction,
    )

    dataloader_multi = create_dataloader_2D_coeff(
        train_data_mo,
        train_coeffs_mo,
        sensor_locations,
        train_timesteps_mo,
        batch_size=32,
        shuffle=False,
        fraction=1,
    )

    dataloader_test_single = create_dataloader_2D(
        test_data,
        sensor_locations,
        test_timesteps,
        batch_size=32,
        shuffle=False,
        fraction=1,
    )

    dataloader_test_multi = create_dataloader_2D_coeff(
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
        # Train a vanilla DeepONet
        vanilla_deeponet, v_loss, v_test_loss = train_deeponet_visualized(
            dataloader_single,
            branch_input_size,
            trunk_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_hidden_layers,
            num_epochs,
            learning_rate,
            test_loader=dataloader_test_single,
            N_sensors=branch_input_size,
            N_timesteps=N_timesteps,
            schedule=False,
        )

        # Save the trained DeepONet
        save_model(
            vanilla_deeponet,
            "singleonet",
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
                "train_duration": train_deeponet_visualized.duration,
            },
        )

        # Train a MODeepONet
        multiple_deeponet, m_loss, m_test_loss = train_multionet_poly_coeff(
            dataloader_multi,
            branch_input_size,
            trunk_input_size - 1,
            hidden_size,
            branch_hidden_layers,
            trunk_hidden_layers,
            output_neurons,
            N_outputs,
            num_epochs * 3,
            learning_rate,
            test_loader=dataloader_test_multi,
            N_sensors=branch_input_size,
            N_timesteps=N_timesteps,
            schedule=False,
        )

        # Save the trained MODeepONet
        save_model(
            multiple_deeponet,
            "multionet",
            {
                "branch_input_size": branch_input_size,
                "trunk_input_size": trunk_input_size,
                "hidden_size": hidden_size,
                "branch_hidden_layers": branch_hidden_layers,
                "trunk_hidden_layers": trunk_hidden_layers,
                "" "num_epochs": num_epochs * 4,
                "learning_rate": learning_rate,
                "N_timesteps": N_timesteps,
                "decay_rate": decay_rate,
                "fraction": 1,
                "num_samples_train": num_samples_train * 5,
                "num_samples_test": num_samples_test,
                "train_duration": train_multionet_poly_coeff.duration,
            },
        )

        # Plot the losses (and save the plot)
        plot_losses(
            (v_loss, v_test_loss, m_loss, m_test_loss),
            (
                "train loss single",
                "test loss single",
                "train loss multi",
                "test loss multi",
            ),
        )

    else:
        # Load the DeepONet
        vanilla_deeponet = load_deeponet(
            "models/02-19/singleonet.pth",
            branch_input_size,
            trunk_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_hidden_layers,
        )

        multiple_deeponet = load_multionet(
            "models/02-19/multionet.pth",
            branch_input_size,
            trunk_input_size - 1,
            hidden_size,
            branch_hidden_layers,
            trunk_hidden_layers,
            output_neurons,
            N_outputs,
            architecture="branch",
        )

    # Test the vanilla DeepONet
    total_loss, v_predictions, v_targets = test_deeponet(
        vanilla_deeponet, dataloader_test_single
    )
    print(f"Average prediction error (DeepONet): {total_loss:.3E}")

    v_predictions = v_predictions.reshape(
        -1, len(sensor_locations), len(test_timesteps)
    )
    v_targets = v_targets.reshape(-1, len(sensor_locations), len(test_timesteps))

    v_errors = np.abs(v_predictions - v_targets)
    v_relative_errors = v_errors / np.abs(v_targets)

    plot_relative_errors_over_time(
        v_relative_errors, title="Relative errors in space (DeepONet)"
    )

    plot_relative_errors_over_time(
        v_relative_errors.transpose(0, 2, 1), title="Relative errors in time (DeepONet)"
    )

    # Plot some vanilla DeepONet results
    heatmap_plot(
        sensor_locations,
        test_timesteps,
        v_targets,
        5,
        v_predictions,
        title="DeepONet results",
    )

    # Test the MODeepONet
    coeff_loss, total_loss, m_predictions, m_targets = test_multionet_poly(
        multiple_deeponet, dataloader_test_multi, sensor_locations
    )

    print(f"Average prediction error (MultiONet): {total_loss:.3E}")

    m_predictions = m_predictions.reshape(
        -1, len(test_timesteps), len(sensor_locations)
    ).transpose(0, 2, 1)
    m_targets = m_targets.reshape(
        -1, len(test_timesteps), len(sensor_locations)
    ).transpose(0, 2, 1)

    m_errors = np.abs(m_predictions - m_targets)
    m_relative_errors = m_errors / np.abs(m_targets)

    plot_relative_errors_over_time(
        m_relative_errors, title="Relative errors in space (MultiONet)"
    )

    plot_relative_errors_over_time(
        m_relative_errors.transpose(0, 2, 1),
        title="Relative errors in time (MultiONet)",
    )

    # Plot some MODeepONet results
    heatmap_plot(
        sensor_locations,
        test_timesteps,
        m_targets,
        5,
        m_predictions,
        title="MultiONet results",
    )

    # Plot the results
    # predictions = v_predictions.reshape(-1, len(sensor_locations), len(test_timesteps))

    # surface_plot(sensor_locations, timesteps, sines, 3, predictions, title="DeepONet results")

    print("Done.")


if __name__ == "__main__":
    run(None)
