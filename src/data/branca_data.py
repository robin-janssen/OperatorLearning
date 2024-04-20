import numpy as np
import h5py

from plotting import compare_datasets_histogram


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

    data_subset = data_new[::5]

    # Save subset as numpy array
    save_path = filepath.replace(".h5", "_subset_3.npy")
    np.save(save_path, data_subset)
    print("Saved subset as numpy array")

    return data_subset


def run(args):
    # filepath = "/export/scratch/rjanssen/branca_data/dataset_LHS.h5"
    # new_path = "/export/scratch/rjanssen/branca_data/dataset_reshaped.h5"
    # new_path = initialize_branca_data(filepath)
    # data_subset = branca_subset(new_path)

    data = np.load("/export/scratch/rjanssen/branca_data/dataset_reshaped_subset.npy")
    data1 = data[:, 0, :]
    data = np.load("/export/scratch/rjanssen/branca_data/dataset_reshaped_subset_2.npy")
    data2 = data[:, 0, :]
    data = np.load("/export/scratch/rjanssen/branca_data/dataset_reshaped_subset_3.npy")
    data3 = data[:, 0, :]

    compare_datasets_histogram(data1, data2, data3, save=True)

    # print("Shape:", data.shape)


if __name__ == "__main__":
    run("args")
    print("Done.")
