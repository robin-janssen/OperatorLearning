# Comparing the performanceof DeepONet trained on the sensor locations of polynomials and the coefficients of polynomials

import numpy as np
import torch
import torch.nn as nn

from data import generate_polynomial_data_coeff
from plotting import plot_functions_only, plot_losses, plot_results
from training import train_deeponet, load_deeponet, save_model
from data import create_dataloader_modified


def test_deeponet_coeff(model, data_loader, sensor_points=[], order=5, coeff=False):
    """
    Test a DeepONet model, with the option to evaluate a DeepONet that predicts the coefficients of the polynomial.

    :param model: A DeepONet model (as instantiated using the DeepONet class).
    :param data_loader: A DataLoader object.
    :param sensor_points: Array of sensor locations in the domain. Required only if coeff=True.
    :param order: Order of the polynomial to be integrated.
    :param coeff: If True, the function will assume that the model returns the coefficients of the polynomial.
    :return: Total loss and predictions.
    """
    model.eval()
    criterion = nn.MSELoss(reduction="sum")

    # Calculate the total number of predictions to pre-allocate the buffer
    total_predictions = sum(len(targets) for _, _, targets in data_loader)
    buffer_index = 0
    all_predictions = np.empty(total_predictions)
    all_targets = np.empty(total_predictions)
    with torch.no_grad():
        for branch_inputs, trunk_inputs, targets in data_loader:
            outputs = model(branch_inputs, trunk_inputs)
            num_predictions = len(targets)
            all_predictions[buffer_index : buffer_index + num_predictions] = (
                outputs.cpu().numpy()
            )
            all_targets[buffer_index : buffer_index + num_predictions] = (
                targets.cpu().numpy()
            )
            buffer_index += num_predictions

    if coeff:
        total_loss = 0
        num_polynomials = all_predictions.shape[0] / (order + 1)
        for i in range(num_polynomials):
            poly_pred = np.poly1d(
                all_predictions[i * (order + 1) : (i + 1) * (order + 1)]
            )
            predictions = poly_pred(sensor_points)
            poly_true = np.poly1d(all_targets[i * (order + 1) : (i + 1) * (order + 1)])
            ground_truth = poly_true(sensor_points)
            loss = criterion(torch.tensor(predictions), torch.tensor(ground_truth))
            total_loss += loss.item()
    else:
        total_loss = (
            criterion(torch.tensor(all_predictions), torch.tensor(all_targets))
            .cpu()
            .numpy()
        )

    return total_loss, all_predictions


if __name__ == "__main__":
    # We would like to compare the performance of DeepONet trained in two different ways:
    # 1) Trained on the sensor locations of polyniomials (up to degree 5)
    # 2) Trained on the coefficients of polynomials (also up to degree 5)

    # We will use the same DeepONet architecture for both cases, but we will need to change the number of inputs for the branch network

    # Lets generate some data
    num_samples_train = 500
    num_samples_test = 100
    sensor_points = np.linspace(0, 1, 101)
    data = generate_polynomial_data_coeff(
        num_samples_train, sensor_points, method="trapezoidal", scale=3
    )
    data_coeff = generate_polynomial_data_coeff(
        num_samples_train, sensor_points, method="trapezoidal", scale=3
    )
    test_data = generate_polynomial_data_coeff(
        num_samples_test, sensor_points, method="trapezoidal", scale=3
    )

    # Generate dataloaders for both datasets
    train_loader = create_dataloader_modified(
        data, sensor_points, batch_size=32, shuffle=True, coeff=False
    )
    train_loader_coeff = create_dataloader_modified(
        data_coeff, sensor_points, batch_size=32, shuffle=True, coeff=True
    )
    test_loader = create_dataloader_modified(
        test_data, sensor_points, batch_size=32, shuffle=False, coeff=False
    )
    test_loader_coeff = create_dataloader_modified(
        test_data, sensor_points, batch_size=32, shuffle=False, coeff=True
    )

    plot_data = np.array([fct for fct, _, _ in data])
    # Now we need to plot some of the data
    plot_functions_only(plot_data[:, :, None], sensor_points, 300)
    # plot_functions_only(np.array(data_coeff), sensor_points, 300)

    # Define two DeepONets, one for each dataset
    branch_hidden_layers = 3
    trunk_hidden_layers = 2
    hidden_size = 40
    trunk_input_size = 1

    # Now we need to train the two DeepONets
    # We will use the same training parameters for both
    epochs = 50
    learning_rate = 0.001
    TRAIN = False

    if TRAIN:
        deeponet, loss = train_deeponet(
            data_loader=train_loader,
            branch_input_size=101,
            trunk_input_size=trunk_input_size,
            hidden_size=hidden_size,
            num_epochs=epochs,
            branch_hidden_layers=branch_hidden_layers,
            trunk_hidden_layers=trunk_hidden_layers,
            learning_rate=learning_rate,
        )
        save_model(
            deeponet,
            "deeponet_restricted",
            {
                "branch_input_size": 101,
                "trunk_input_size": trunk_input_size,
                "hidden_size": hidden_size,
                "branch_hidden_layers": branch_hidden_layers,
                "trunk_hidden_layers": trunk_hidden_layers,
                "num_epochs": epochs,
                "learning_rate": learning_rate,
                "num_samples_train": num_samples_train,
                "num_samples_test": num_samples_test,
            },
        )

        deeponet_coeff, loss_coeff = train_deeponet(
            data_loader=train_loader_coeff,
            branch_input_size=6,
            trunk_input_size=trunk_input_size,
            hidden_size=hidden_size,
            num_epochs=epochs,
            branch_hidden_layers=branch_hidden_layers,
            trunk_hidden_layers=trunk_hidden_layers,
            learning_rate=learning_rate,
        )
        save_model(
            deeponet_coeff,
            "deeponet_restricted_coeff",
            {
                "branch_input_size": 6,
                "trunk_input_size": trunk_input_size,
                "hidden_size": hidden_size,
                "branch_hidden_layers": branch_hidden_layers,
                "trunk_hidden_layers": trunk_hidden_layers,
                "num_epochs": epochs,
                "learning_rate": learning_rate,
                "num_samples_train": num_samples_train,
                "num_samples_test": num_samples_test,
            },
        )

        plot_losses((loss, loss_coeff), ("Standard", "Coefficients"))

    else:
        deeponet = load_deeponet(
            "models/02-06/deeponet_restricted.pth",
            101,
            trunk_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_hidden_layers,
        )
        deeponet_coeff = load_deeponet(
            "models/02-06/deeponet_restricted_coeff.pth",
            6,
            trunk_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_hidden_layers,
        )

    # Now we need to test the two DeepONets
    # We will use the same testing parameters for both

    standard_error, standard_prediction = test_deeponet_coeff(deeponet, test_loader)
    coeff_error, coeff_prediction = test_deeponet_coeff(
        deeponet_coeff, test_loader_coeff
    )

    print("Standard DeepONet error: ", standard_error)
    print("DeepONet with coefficients error: ", coeff_error)

    standard_prediction = standard_prediction.reshape(-1, len(sensor_points))
    coeff_prediction = coeff_prediction.reshape(-1, len(sensor_points))

    fct_values, ground_truth, coefficients = [], [], []
    for i in range(10):
        fct_value, gt, coeff = test_data[i]
        fct_values.append(fct_value)
        ground_truth.append(gt)
        coefficients.append(coeff)

    plot_results(
        "Standard DeepONet",
        sensor_points,
        fct_values,
        ground_truth,
        sensor_points,
        standard_prediction,
    )
    plot_results(
        "DeepONet with Coefficients",
        sensor_points,
        fct_values,
        ground_truth,
        sensor_points,
        coeff_prediction,
    )

    print("Finished training")
