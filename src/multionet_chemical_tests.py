import numpy as np
from torchinfo import summary

from plotting import (
    plot_chemical_examples,
    plot_chemicals_comparative,
    plot_losses,
    plot_chemical_results,
)
from training import (
    create_dataloader_chemicals,
    train_multionet_chemical,
    test_deeponet,
    load_multionet,
)

from utils import save_model, load_chemical_data


if __name__ == "__main__":

    TRAIN = True
    CONT = False
    VIS = False
    branch_input_size = 29
    trunk_input_size = 1
    hidden_size = 100
    branch_hidden_layers = 2
    trunk_hidden_layers = 2
    num_epochs = 100
    learning_rate = 3e-4
    fraction = 1
    output_neurons = 290  # number of neurons in the last layer of MODeepONet
    N_outputs = 29  # number of outputs of MODeepONet

    data = load_chemical_data("data/dataset1")
    data_shape = data.shape
    print(f"Data shape: {data_shape}")

    # Use only the amount of each chemical, not the gradients
    data = data[:, :, :29]

    N_timesteps = data.shape[1]

    # Split the data into training and testing (80/20)
    train_data = data[: int(0.8 * data.shape[0])]
    test_data = data[int(0.8 * data.shape[0]) :]

    chemicals = "H, H+, H2, H2+, H3+, O, O+, OH+, OH, O2, O2+, H2O, H2O+, H3O+, C, C+, CH, CH+, CH2, CH2+, CH3, CH3+, CH4, CH4+, CH5+, CO, CO+, HCO+, He, He+, E"

    extracted_chemicals = chemicals.split(", ")

    if VIS:
        plot_chemical_examples(data, extracted_chemicals)
        plot_chemicals_comparative(data, extracted_chemicals)
    timesteps = np.arange(data.shape[1])

    dataloader_train = create_dataloader_chemicals(
        train_data, timesteps, fraction=1, batch_size=32, shuffle=True
    )

    dataloader_test = create_dataloader_chemicals(
        test_data, timesteps, fraction=1, batch_size=32, shuffle=False
    )

    data_example = next(iter(dataloader_train))

    if TRAIN:
        multionet, train_loss, test_loss = train_multionet_chemical(
            dataloader_train,
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

        # Save the MulitoNet
        save_model(
            multionet,
            "multionet_chemical",
            {
                "branch_input_size": branch_input_size,
                "trunk_input_size": trunk_input_size,
                "hidden_size": hidden_size,
                "branch_hidden_layers": branch_hidden_layers,
                "trunk_hidden_layers": trunk_hidden_layers,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "N_timesteps": N_timesteps,
                "fraction": fraction,
                "num_samples_train": train_data.shape[0],
                "num_samples_test": test_data.shape[0],
                "train_duration": train_multionet_chemical.duration,
                "architecture": "both",
            },
        )

        plot_losses(
            (train_loss,), ("train loss",), "Loss (MultiONet for chemical data)"
        )

    else:
        multionet_coeff = load_multionet(
            "models/27-02/multionet_chemical.pth",
            branch_input_size,
            trunk_input_size,
            hidden_size,
            branch_hidden_layers,
            trunk_hidden_layers,
            output_neurons,
            N_outputs,
            architecture="both",
        )

    # Print the model summary
    summary(multionet_coeff, input_size=[(32, 29), (32, 1)])

    # Predict the test data
    error, predictions, ground_truth = test_deeponet(multionet_coeff, dataloader_test)

    predictions = predictions.reshape(-1, N_timesteps, N_outputs)
    ground_truth = ground_truth.reshape(-1, N_timesteps, N_outputs)

    # Plot the results
    plot_chemical_results(
        predictions, ground_truth, extracted_chemicals, num_chemicals=5
    )

    print("Done!")
