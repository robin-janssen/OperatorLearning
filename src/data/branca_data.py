import numpy as np
import h5py

from plotting import compare_datasets_histogram
from .data_utils import train_test_split


def initialize_branca_data(filepath):
    n_ic = 256
    n_density = 64
    n_temp = 64
    n_time = 16  # nicht 17!
    n_species = 10
    n_rad = 64
    new_path = "/export/scratch/rjanssen/branca_data/dataset_reshaped.h5"

    file = h5py.File(filepath, "r")
    print(list(file.keys()))
    data = file["data"]
    print("Shape:", data.shape)
    with h5py.File(new_path, "w") as file:
        data_new = file.create_dataset(
            "data_new",
            (int(n_ic * n_density * n_temp * n_rad), n_time, n_species),
            dtype="f",
        )
        for i in range(10):
            data_new[:, :, i] = data[:, i].reshape(
                (int(n_ic * n_density * n_temp * n_rad), n_time)
            )
            print(f"Reshaped data for {i}/10 chemicals")

    return new_path


# Example usage:
# Assume dataset1, dataset2, and dataset3 are defined with the appropriate shapes
# compare_datasets_histogram(dataset1, dataset2, dataset3)


def branca_subset(filepath):
    file = h5py.File(filepath, "r")
    data_new = file["data_new"]
    print("Shape:", data_new.shape)

    # n_samples = data_new.shape[0]
    # n_samples_subset = int(n_samples * fraction)
    # print(f"Selecting {n_samples_subset} samples from {n_samples} samples")
    # indices = np.random.choice(n_samples, n_samples_subset, replace=False)
    # print("Indices:", indices[:10])
    # indices = np.sort(indices)
    # print("Sorted indices:", indices[:10])
    # data_subset = data_new[indices]
    # print("Shape of subset:", data_subset.shape)

    data_subset = data_new[::300]

    # Save subset as numpy array
    save_path = filepath.replace(".h5", "_subset_0.npy")
    np.save(save_path, data_subset)
    print("Saved subset as numpy array")

    return data_subset


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
    # print("Smallest non-zero value:", np.min(np.abs(train_data[train_data != 0])))

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
    # filepath = "/export/scratch/rjanssen/branca_data/dataset_LHS.h5"
    # new_path = "/export/scratch/rjanssen/branca_data/dataset_reshaped.h5"
    # new_path = initialize_branca_data(filepath)
    # data_subset = branca_subset(new_path)

    data = np.load("/export/scratch/rjanssen/branca_data/dataset_reshaped_subset_0.npy")
    data0 = data[:, 0, :]
    # data = np.load("/export/scratch/rjanssen/branca_data/dataset_reshaped_subset.npy")
    # data1 = data[:, 0, :]
    data = np.load("/export/scratch/rjanssen/branca_data/dataset_reshaped_subset_2.npy")
    data2 = data[:, 0, :]
    data = np.load("/export/scratch/rjanssen/branca_data/dataset_reshaped_subset_3.npy")
    data3 = data[:, 0, :]

    compare_datasets_histogram(data0, data2, data3, save=True)

    # print("Shape:", data.shape)

    # data = np.load("/export/scratch/rjanssen/branca_data/dataset_reshaped_subset_0.npy")

    # analyze_branca_data(data)

    # train_data, test_data, timesteps = prepare_branca_data(
    #     data, train_cut=50000, test_cut=10000
    # )

    # print("Shape:", test_data.shape)


if __name__ == "__main__":
    run("args")
    print("Done.")
