import pickle
import numpy as np
from scipy.interpolate import interp1d

from utils import get_project_path


def load_fc_spectra(filename="spectral-data-free-cooling-large.pkl", interpolate=True):
    """
    Load the free cooling spectral data from a pickle file.

    :param filename: The name of the pickle file containing the spectral data.
    :param interpolate: Whether to interpolate the data.
    :return: The spectral data as a numpy array.
    """

    location = "data/free_cooling/" + filename
    load_path = get_project_path(location)

    with open(load_path, "rb") as fp:
        raw_data = pickle.load(fp)
        print("dictionary read successfully from file")

    print("data keys: ", raw_data.keys())
    data = raw_data["data"]
    data = data.reshape(-1, 11, 2, 100, order="F")
    timesteps = raw_data["times"]
    sl = raw_data["slopes_low"]
    sh = raw_data["slopes_high"]
    w = raw_data["widths"]
    if interpolate:
        p_values = data[0, 0, 0, :]  # x-values from timestep 0
        cleaned_data = np.zeros_like(data)
        cleaned_data[:, 0, :, :] = data[:, 0, :, :]  # No need to clean timestep 0

        for t in range(1, data.shape[1]):  # Start from timestep 1
            # Calculate mask for non-zero p-values
            valid_mask = data[0, t, 0, :] > 0
            x_values = data[0, t, 0, valid_mask]  # Use valid x-values for interpolation

            for i in range(data.shape[0]):
                y_values = data[i, t, 1, valid_mask]
                if np.any(valid_mask) and not np.all(
                    np.isnan(y_values)
                ):  # Check for any valid y-values
                    interp_func = interp1d(
                        x_values, y_values, bounds_error=False, fill_value="extrapolate"
                    )
                    cleaned_y_values = interp_func(p_values)
                    cleaned_data[i, t, 0, :] = p_values
                    cleaned_data[i, t, 1, :] = cleaned_y_values

        # Ensure all p_values for different timesteps and samples are the same
        assert np.all(cleaned_data[:, :, 0, :] == cleaned_data[0, 0, 0, :])

    else:
        cleaned_data = data

    # Check for and print indices of any negative values in the data
    negative_indices = np.where(cleaned_data < 0)

    # Find the lowest index with a negative value
    ind_low = np.min(negative_indices[3])
    cleaned_data = cleaned_data[:, :, :, :ind_low]

    negative_indices = np.where(cleaned_data < 0)
    if len(negative_indices[0]) > 0:
        print("Negative values in the data:")
        print(negative_indices)
    else:
        print("No negative values in the cleaned data.")

    # Take the log of the data
    cleaned_data = np.log10(cleaned_data)

    # Use equispaced timesteps
    print(f"Timesteps: {timesteps}")
    print(f"Data shape: {cleaned_data.shape}")
    return cleaned_data, timesteps, sl, sh, w
