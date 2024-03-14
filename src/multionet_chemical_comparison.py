import numpy as np
from torchinfo import summary

from chemicals import chemicals
from plotting import (
    plot_chemical_examples,
    plot_chemicals_comparative,
    plot_losses,
    plot_chemical_results,
    plot_chemical_errors,
    plot_relative_errors_over_time,
    plot_functions_only,
)
from training import (
    create_dataloader_chemicals,
    train_multionet_chemical,
    test_deeponet,
    load_multionet,
)
from utils import save_model, load_chemical_data, read_yaml_config


if __name__ == "__main__":

    TRAIN = False
    VIS = False
    USE_MASS_CONSERVATION = True
    pretrained_model_path = None  # "models/02-28/multionet_chemical_500.pth"
    branch_input_size = 29
    trunk_input_size = 1
    hidden_size = 100
    branch_hidden_layers = 2
    trunk_hidden_layers = 2
    num_epochs = 200
    learning_rate = 3e-4
    fraction = 1
    output_neurons = 290  # number of neurons in the last layer of MODeepONet
    N_outputs = 29  # number of outputs of MODeepONet
    architecture = "both"  # "both", "branch", "trunk"
    device = "mps"  # "cpu", "mps"

    if USE_MASS_CONSERVATION:
        from chemicals import masses
    else:
        masses = None

    # data = load_chemical_data("data/dataset100")
    data = load_chemical_data("data/dataset1000")
    data_shape = data.shape
    print(f"Data shape: {data_shape}")

    # Use only the amount of each chemical, not the gradients
    data = data[:, :, :29]

    N_timesteps = data.shape[1]
    timesteps = np.arange(data.shape[1])

    # Split the data into training and testing (80/20)
    # train_data = data[: int(0.8 * data.shape[0])]
    # test_data = data[int(0.8 * data.shape[0]) :]
    train_data = data[:500]
    test_data = data[500:550]

    extracted_chemicals = chemicals.split(", ")

    if VIS:
        plot_chemical_examples(data, extracted_chemicals)
        plot_chemicals_comparative(data, extracted_chemicals)

    dataloader_train = create_dataloader_chemicals(
        train_data, timesteps, fraction=1, batch_size=32, shuffle=True
    )

    dataloader_test = create_dataloader_chemicals(
        test_data, timesteps, fraction=1, batch_size=32, shuffle=False
    )

    data_example = next(iter(dataloader_train))

    multionet_standard = load_multionet(
        "models/02-28/multionet_chemical_500.pth",
        branch_input_size,
        trunk_input_size,
        hidden_size,
        branch_hidden_layers,
        trunk_hidden_layers,
        output_neurons,
        N_outputs,
        architecture="both",
        device=device,
    )

    multionet_massloss = load_multionet(
        "models/03-09/multionet_chemical_fine2_200e.pth",
        branch_input_size,
        trunk_input_size,
        hidden_size=100,
        branch_hidden_layers=5,
        trunk_hidden_layers=5,
        output_neurons=290,
        N_outputs=29,
        architecture="both",
        device=device,
    )

    # Test the DeepONet with standard loss
    loss_st, predictions_st, ground_truth_st = test_deeponet(
        multionet_standard, dataloader_test, timing=True
    )
    print(f"Average MSE loss (MultiONet): {loss_st:.3E}")

    # Calculate the prediction errors
    predictions_st = predictions_st.reshape(-1, N_timesteps, N_outputs)
    ground_truth_st = ground_truth_st.reshape(-1, N_timesteps, N_outputs)
    errors_st = np.abs(predictions_st - ground_truth_st)
    relative_errors_st = errors_st / np.abs(ground_truth_st)
    errors_mean_st = np.mean(errors_st, axis=(0, 2))
    errors_median_st = np.median(errors_st, axis=(0, 2))

    plot_relative_errors_over_time(
        relative_errors_st, "Relative errors over time (First MultiONet for Chemicals)"
    )

    # Test the DeepONet with mass loss
    loss_ml, predictions_ml, ground_truth_ml = test_deeponet(
        multionet_massloss, dataloader_test, timing=True
    )
    print(f"Average MSE loss (MultiONet with mass loss): {loss_ml:.3E}")

    # Calculate the prediction errors
    predictions_ml = predictions_ml.reshape(-1, N_timesteps, N_outputs)
    ground_truth_ml = ground_truth_ml.reshape(-1, N_timesteps, N_outputs)
    errors_ml = np.abs(predictions_ml - ground_truth_ml)
    relative_errors_ml = errors_ml / np.abs(ground_truth_ml)
    errors_mean_ml = np.mean(errors_ml, axis=(0, 2))
    errors_median_ml = np.median(errors_ml, axis=(0, 2))

    plot_relative_errors_over_time(
        relative_errors_ml, "Relative errors over time (New MultiONet for Chemicals)"
    )

    errors = np.stack(
        (errors_mean_st, errors_median_st, errors_mean_ml, errors_median_ml), axis=1
    )
    labels = (
        "First model mean",
        "First model median",
        "New model mean",
        "New model median",
    )

    plot_chemical_errors(
        errors,
        labels,
        title="Prediction errors (comparing two MultiONets for Chemicals)",
    )

    # Plot average
