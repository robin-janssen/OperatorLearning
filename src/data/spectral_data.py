import pickle

from utils import get_project_path


def load_fc_spectra(filename="spectral-data-free-cooling-large.pkl"):
    """
    Load the free cooling spectral data from a pickle file.

    :param filename: The name of the pickle file containing the spectral data.
    :return: The spectral data as a numpy array.
    """

    location = "data/free_cooling/" + filename
    load_path = get_project_path(location)

    with open(load_path, "rb") as fp:
        raw_data = pickle.load(fp)
        print("dictionary read successfully from file")

    data = raw_data["data"]
    return data
