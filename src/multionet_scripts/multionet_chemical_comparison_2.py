# Comparison of the performance of the MultiONet with different fine-tuning strategies
# The first model is the original MultiONet trained with the initial settings.
# The second model is the result of some rough fine-tuning of the architecture and hyperparameters.
# The third model is the result of a more extensive fine-tuning.

import numpy as np
from data.osu_chemicals import chemicals
from plotting import (
    plot_chemical_examples,
    plot_chemicals_comparative,
    plot_chemical_results,
    plot_chemical_results_and_errors,
    plot_chemical_errors,
    plot_relative_errors_over_time,
)
from data import create_dataloader_chemicals, load_chemical_data
from training import test_deeponet, load_multionet


def run(args):

    # TRAIN = False
    # VIS = False
    # USE_MASS_CONSERVATION = True
    # pretrained_model_path = None  # "models/02-28/multionet_chemical_500.pth"
    # branch_input_size = 29
    # trunk_input_size = 1
    # hidden_size = 100
    # branch_hidden_layers = 2
    # trunk_hidden_layers = 2
    # num_epochs = 200
    # learning_rate = 3e-4
    # fraction = 1
    # output_neurons = 290  # number of neurons in the last layer of MODeepONet
    # N_outputs = 29  # number of outputs of MODeepONet
    # architecture = "both"  # "both", "branch", "trunk"
    # device = "mps"  # "cpu", "mps"

    VIS = args.vis
    branch_input_size = args.branch_input_size
    trunk_input_size = args.trunk_input_size
    hidden_size = args.hidden_size
    branch_hidden_layers = args.branch_hidden_layers
    trunk_hidden_layers = args.trunk_hidden_layers
    output_neurons = args.output_neurons
    N_outputs = args.n_outputs
    device = args.device

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
    test_data = data[500:550]

    extracted_chemicals = chemicals.split(", ")

    if VIS:
        plot_chemical_examples(data, extracted_chemicals)
        plot_chemicals_comparative(data, extracted_chemicals)

    dataloader_test = create_dataloader_chemicals(
        test_data, timesteps, fraction=1, batch_size=32, shuffle=False
    )

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

    multionet_fine = load_multionet(
        "models/03-08/multionet_chemical_fine_200e.pth",
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

    multionet_fine_2 = load_multionet(
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
        relative_errors_st, "Relative errors over time (First MultiONet)"
    )

    # Test the DeepONet with mass loss
    loss_fine, predictions_fine, ground_truth_fine = test_deeponet(
        multionet_fine, dataloader_test, timing=True
    )
    print(f"Average MSE loss (Finetuned MultiONet): {loss_fine:.3E}")

    # Calculate the prediction errors
    predictions_fine = predictions_fine.reshape(-1, N_timesteps, N_outputs)
    ground_truth_fine = ground_truth_fine.reshape(-1, N_timesteps, N_outputs)
    errors_fine = np.abs(predictions_fine - ground_truth_fine)
    relative_errors_fine = errors_fine / np.abs(ground_truth_fine)
    errors_mean_ml = np.mean(errors_fine, axis=(0, 2))
    errors_median_ml = np.median(errors_fine, axis=(0, 2))

    plot_relative_errors_over_time(
        relative_errors_fine, "Relative errors over time (Finetuned MultiONet)"
    )

    loss_fine_2, predictions_fine_2, ground_truth_fine_2 = test_deeponet(
        multionet_fine_2, dataloader_test, timing=True
    )

    print(f"Average MSE loss (MultiONet with more fine-tuning): {loss_fine_2:.3E}")

    # Calculate the prediction errors
    predictions_fine_2 = predictions_fine_2.reshape(-1, N_timesteps, N_outputs)
    ground_truth_fine_2 = ground_truth_fine_2.reshape(-1, N_timesteps, N_outputs)
    errors_fine_2 = np.abs(predictions_fine_2 - ground_truth_fine_2)
    relative_errors_fine_2 = errors_fine_2 / np.abs(ground_truth_fine_2)
    errors_mean_ml_2 = np.mean(errors_fine_2, axis=(0, 2))
    errors_median_ml_2 = np.median(errors_fine_2, axis=(0, 2))

    plot_relative_errors_over_time(
        relative_errors_fine_2,
        "Relative errors over time (MultiONet with more fine-tuning)",
    )

    mean_errors = np.stack(
        (
            errors_mean_st,
            errors_mean_ml,
            errors_mean_ml_2,
        ),
        axis=1,
    )

    median_errors = np.stack(
        (
            errors_median_st,
            errors_median_ml,
            errors_median_ml_2,
        ),
        axis=1,
    )

    labels = (
        "First model",
        "Finetuned model",
        "Finetuned model 2",
    )

    plot_chemical_errors(
        mean_errors,
        labels,
        title="Mean errors",
    )

    plot_chemical_errors(
        median_errors,
        labels,
        title="Median errors",
    )

    all_predictions = (predictions_st, predictions_fine, predictions_fine_2)

    model_names = ("First model", "Finetuned model", "Finetuned model 2")

    plot_chemical_results(
        predictions=all_predictions,
        ground_truth=ground_truth_st,
        names=extracted_chemicals,
        num_chemicals=4,
        model_names=model_names,
    )

    plot_chemical_results_and_errors(
        predictions=all_predictions,
        ground_truth=ground_truth_st,
        names=extracted_chemicals,
        num_chemicals=4,
        model_names=model_names,
    )
