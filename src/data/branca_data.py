# import numpy as np
import h5py


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


def load_branca_data(filepath):
    file = h5py.File(filepath, "r")
    data_new = file["data_new"]
    print("Shape:", data_new.shape)


def run(args):
    filepath = "/export/scratch/rjanssen/branca_data/dataset_LHS.h5"

    new_path = initialize_branca_data(filepath)

    load_branca_data(new_path)
