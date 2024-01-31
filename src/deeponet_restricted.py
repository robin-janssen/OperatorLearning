import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from datagen import numerical_integration, plot_functions_only
from deeponet_training import train_deeponet, plot_losses, plot_results, load_deeponet


# Modified generate_polynomial_data function
def generate_polynomial_data_coeff(num_samples, sensor_points, scale=1, method='trapezoidal', order=5):
    '''Generate a polynomial dataset with random coefficients as in generate_polynomial_data,
    but return the coefficients of the input polynomial.
    '''
    data = []
    for _ in range(num_samples):
        # Random coefficients for a cubic polynomial
        coefficients = np.random.uniform(-scale, scale, order + 1)
        # coefficients_1 = np.random.uniform(-3*scale, 3*scale, 3)
        # coefficients_2 = np.random.uniform(-scale, scale, 3)
        # coefficients = np.concatenate((coefficients_1, coefficients_2))
        polynomial = np.poly1d(coefficients)
        poly = polynomial(sensor_points)

        # Compute antiderivative
        # antiderivative = polynomial.integ()
        antiderivative = numerical_integration(poly, sensor_points, method=method)

        data.append((poly, antiderivative, coefficients))

    return data


def create_dataloader_modified(data, sensor_points, batch_size=32, shuffle=False, coeff=False):
    """
    Create a DataLoader for DeepONet.

    :param data: List of tuples (function_values, antiderivative_values, coefficients).
    :param sensor_points: Array of sensor locations in the domain.
    :param batch_size: Batch size for the DataLoader.
    :return: A DataLoader object.
    """
    branch_inputs = []
    trunk_inputs = []
    targets = []

    if coeff:
        for function_values, antiderivative_values, coefficients in data:
            for i in range(len(sensor_points)):
                branch_inputs.append(coefficients)
                trunk_inputs.append([sensor_points[i]])
                targets.append(antiderivative_values[i])
    else:
        for function_values, antiderivative_values, coefficients in data:
            for i in range(len(sensor_points)):
                branch_inputs.append(function_values)
                trunk_inputs.append([sensor_points[i]])
                targets.append(antiderivative_values[i])

    # Convert to PyTorch tensors
    branch_inputs_tensor = torch.tensor(np.array(branch_inputs), dtype=torch.float32)
    trunk_inputs_tensor = torch.tensor(trunk_inputs, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(branch_inputs_tensor, trunk_inputs_tensor, targets_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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
    criterion = nn.MSELoss(reduction='sum')

    # Calculate the total number of predictions to pre-allocate the buffer
    total_predictions = sum(len(targets) for _, _, targets in data_loader)
    buffer_index = 0
    all_predictions = np.empty(total_predictions)
    all_targets = np.empty(total_predictions)
    with torch.no_grad():
        for branch_inputs, trunk_inputs, targets in data_loader:
            outputs = model(branch_inputs, trunk_inputs)
            num_predictions = len(targets)
            all_predictions[buffer_index:buffer_index + num_predictions] = outputs.cpu().numpy()
            all_targets[buffer_index:buffer_index + num_predictions] = targets.cpu().numpy()
            buffer_index += num_predictions

    if coeff:
        total_loss = 0
        num_polynomials = all_predictions.shape[0] / (order + 1)
        for i in range(num_polynomials):
            poly_pred = np.poly1d(all_predictions[i * (order + 1):(i + 1) * (order + 1)])
            predictions = poly_pred(sensor_points)
            poly_true = np.poly1d(all_targets[i * (order + 1):(i + 1) * (order + 1)])
            ground_truth = poly_true(sensor_points)
            loss = criterion(torch.tensor(predictions), torch.tensor(ground_truth))
            total_loss += loss.item()
    else:
        total_loss = criterion(torch.tensor(all_predictions), torch.tensor(all_targets)).cpu().numpy()

    return total_loss, all_predictions


if __name__ == "__main__":
    # We would like to compare the performance of DeepONet trained in two different ways:
    # 1) Trained on the sensor locations of polyniomials (up to degree 5)
    # 2) Trained on the coefficients of polynomials (also up to degree 5)

    # We will use the same DeepONet architecture for both cases, but we will need to change the number of inputs for the branch network

    # Lets generate some data
    num_samples = 500
    sensor_points = np.linspace(0, 1, 101)
    data = generate_polynomial_data_coeff(num_samples, sensor_points, method='trapezoidal', scale=3)
    data_coeff = generate_polynomial_data_coeff(num_samples, sensor_points, method='trapezoidal', scale=3)
    test_data = generate_polynomial_data_coeff(num_samples, sensor_points, method='trapezoidal', scale=3)

    # Generate dataloaders for both datasets
    train_loader = create_dataloader_modified(data, sensor_points, batch_size=32, shuffle=True, coeff=False)
    train_loader_coeff = create_dataloader_modified(data_coeff, sensor_points, batch_size=32, shuffle=True, coeff=True)
    test_loader = create_dataloader_modified(test_data, sensor_points, batch_size=32, shuffle=False, coeff=False)
    test_loader_coeff = create_dataloader_modified(test_data, sensor_points, batch_size=32, shuffle=False, coeff=True)

    # Now we need to plot some of the data
    plot_functions_only(data, sensor_points, 300)
    plot_functions_only(data_coeff, sensor_points, 300)

    # Define two DeepONets, one for each dataset
    branch_hidden_layers = 3
    trunk_hidden_layers = 2
    hidden_size = 40
    trunk_input_size = 1

    # Now we need to train the two DeepONets
    # We will use the same training parameters for both
    epochs = 20
    learning_rate = 0.001
    TRAIN = True

    if TRAIN:
        deeponet, loss = train_deeponet(
            data_loader=train_loader,
            branch_input_size=101,
            trunk_input_size=trunk_input_size,
            hidden_size=hidden_size,
            num_epochs=epochs,
            branch_hidden_layers=branch_hidden_layers,
            trunk_hidden_layers=trunk_hidden_layers,
            learning_rate=learning_rate
        )
        torch.save(deeponet.state_dict(), 'deeponet_restricted.pth')

        deeponet_coeff, loss_coeff = train_deeponet(
            data_loader=train_loader_coeff,
            branch_input_size=6,
            trunk_input_size=trunk_input_size,
            hidden_size=hidden_size,
            num_epochs=epochs,
            branch_hidden_layers=branch_hidden_layers,
            trunk_hidden_layers=trunk_hidden_layers,
            learning_rate=learning_rate
        )
        torch.save(deeponet_coeff.state_dict(), 'deeponet_restricted_coeff.pth')

        plot_losses((loss, loss_coeff), ("Standard", "Coefficients"))

    else:
        deeponet = load_deeponet('deeponet_restricted.pth', 101, trunk_input_size, hidden_size, branch_hidden_layers, trunk_hidden_layers)
        deeponet_coeff = load_deeponet('deeponet_restricted_coeff.pth', 6, trunk_input_size, hidden_size, branch_hidden_layers, trunk_hidden_layers)

    # Now we need to test the two DeepONets
    # We will use the same testing parameters for both

    standard_error, standard_prediction = test_deeponet_coeff(deeponet, test_loader)
    coeff_error, coeff_prediction = test_deeponet_coeff(deeponet_coeff, test_loader_coeff)

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

    plot_results('Standard DeepONet', sensor_points, fct_values, ground_truth, sensor_points, standard_prediction)
    plot_results('DeepONet with Coefficients', sensor_points, fct_values, ground_truth, sensor_points, coeff_prediction)

    print("Finished training")
