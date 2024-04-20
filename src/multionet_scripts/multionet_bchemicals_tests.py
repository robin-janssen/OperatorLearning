# Script to train a MultioNet model for the Branca chemicals dataset.
import numpy as np

from data import create_dataloader_chemicals, train_test_split
from training import (
    BChemicalTrainConfig,
    train_multionet_chemical,
    save_model,
    test_deeponet,
    load_multionet,
)
from plotting import (
    plot_chemical_examples,
    plot_chemicals_comparative,
    plot_relative_errors_over_time,
    plot_chemical_results,
    plot_losses,
)


def prepare_branca_data(data, train_cut=300, test_cut=100):
    """
    Prepare the Priestley data for training.
    """
    timesteps = np.linspace(0, 15, 16)
    np.random.shuffle(data)
    train_data, test_data = train_test_split(data, train_fraction=0.8)
    train_data = train_data[:train_cut, :, :]
    test_data = test_data[:test_cut, :, :]

    # Find smallest non-zero value in the dataset
    print("Smallest non-zero value:", np.min(np.abs(train_data[train_data != 0])))

    # Find zero values in the dataset and replace them with the smallest non-zero value
    zero_indices = np.where(train_data == 0)
    train_data[zero_indices] = np.min(np.abs(train_data[train_data != 0]))
    zero_indices = np.where(test_data == 0)
    test_data[zero_indices] = np.min(np.abs(test_data[test_data != 0]))

    train_data = np.log10(train_data)
    test_data = np.log10(test_data)
    return train_data, test_data, timesteps


def analyze_branca_data(data):
    """
    Analyze the Branca data.
    """
    # print("Data min:", np.min(data))
    # print("Data max:", np.max(data))
    # print("Data mean:", np.mean(data))
    # print("Data std:", np.std(data))
    # initial_conditions = data[:, 0, :]
    # print("Initial conditions min:", np.min(initial_conditions))
    # print("Initial conditions max:", np.max(initial_conditions))
    # print("Initial conditions mean:", np.mean(initial_conditions))
    # print("Initial conditions std:", np.std(initial_conditions))
    # for i in range(10):
    #     initial_cond_chem = initial_conditions[:, i]
    #     print(f"Chemical {i} min:", np.min(initial_cond_chem))
    #     print(f"Chemical {i} max:", np.max(initial_cond_chem))
    #     print(f"Chemical {i} mean:", np.mean(initial_cond_chem))
    #     print(f"Chemical {i} std:", np.std(initial_cond_chem))

    # Write all prints to a file
    with open("branca_subset_3_analysis.txt", "w") as f:
        print("Data min:", np.min(data), file=f)
        print("Data max:", np.max(data), file=f)
        print("Data mean:", np.mean(data), file=f)
        print("Data std:", np.std(data), file=f)
        print("\n")
        initial_conditions = data[:, 0, :]
        print("Initial conditions min:", np.min(initial_conditions), file=f)
        print("Initial conditions max:", np.max(initial_conditions), file=f)
        print("Initial conditions mean:", np.mean(initial_conditions), file=f)
        print("Initial conditions std:", np.std(initial_conditions), file=f)
        print("\n")
        for i in range(10):
            initial_cond_chem = initial_conditions[:, i]
            print(f"Chemical {i} min:", np.min(initial_cond_chem), file=f)
            print(f"Chemical {i} max:", np.max(initial_cond_chem), file=f)
            print(f"Chemical {i} mean:", np.mean(initial_cond_chem), file=f)
            print(f"Chemical {i} std:", np.std(initial_cond_chem), file=f)
            print("\n")


def run(args):

    config = BChemicalTrainConfig()
    # config.device = args.device
    TRAIN = True
    args.vis = True

    # Load the data

    data = np.load("/export/scratch/rjanssen/branca_data/dataset_reshaped_subset_3.npy")
    analyze_branca_data(data)
    train_data, test_data, timesteps = prepare_branca_data(
        data, train_cut=500000, test_cut=100000
    )

    print(f"Time steps: {timesteps}")

    print(
        f"Loaded chemical train/test data with shape: {train_data.shape}/{test_data.shape}"
    )

    if args.vis:
        plot_chemical_examples(
            train_data,
            num_chemicals=10,
            save=True,
            title="Chemical Examples (Branca Data)",
        )
        plot_chemicals_comparative(
            train_data,
            num_chemicals=10,
            save=True,
            title="Chemical Comparison (Branca Data)",
        )

    dataloader_train = create_dataloader_chemicals(
        train_data, timesteps, fraction=1, batch_size=config.batch_size, shuffle=True
    )

    dataloader_test = create_dataloader_chemicals(
        test_data, timesteps, fraction=1, batch_size=config.batch_size, shuffle=False
    )

    if TRAIN:
        multionet, train_loss, test_loss = train_multionet_chemical(
            config, dataloader_train, dataloader_test
        )

        # Save the MulitONet
        save_model(
            multionet,
            "multionet_bchemicals_3",
            config,
            train_loss=train_loss,
            test_loss=test_loss,
        )

    else:
        model_path = "models/04-18/multionet_bchemicals_3"
        multionet, train_loss, test_loss = load_multionet(
            config, config.device, model_path
        )

    plot_losses(
        (train_loss, test_loss),
        ("Train loss", "Test loss"),
        "Losses (MultiONet on Branca Data)",
        save=True,
    )

    # average_error, predictions, ground_truth = test_deeponet(
    #     multionet, dataloader_test, N_timesteps=config.N_timesteps
    # )

    average_error, predictions, ground_truth = test_deeponet(
        multionet,
        dataloader_test,
        N_timesteps=config.N_timesteps,
        reshape=True,
    )

    print(f"Average prediction error: {average_error:.3E}")

    errors = np.abs(predictions - ground_truth)
    relative_errors = errors / np.abs(ground_truth)

    plot_relative_errors_over_time(
        relative_errors,
        "Relative errors over time (MultiONet for Chemicals)",
        save=True,
    )

    plot_chemical_results(
        predictions=predictions,
        ground_truth=ground_truth,
        # names=extracted_chemicals,
        num_chemicals=10,
        model_names="MultiONet",
        save=True,
    )

    # plot_chemical_results_and_errors(
    #     predictions=predictions,
    #     ground_truth=ground_truth,
    #     # names=extracted_chemicals,
    #     num_chemicals=4,
    #     model_names="MultiONet",
    # )

    print("Done!")
