import numpy as np
import os

file_path = "data/test_outputs/"
# Parse all the files in the directory
files = os.listdir(file_path)
# Iterate over all the files
all_data = np.zeros((len(files), 100, 58))
for i, file in enumerate(files):
    # Read the file
    data = np.loadtxt(file_path + file)
    # Perform the test
    all_data[i] = data

print(all_data.shape)

print("Done!")
