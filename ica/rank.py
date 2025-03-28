import json
import numpy as np

# Specify the path to your processed JSON file
json_file = 'HS_P1_S1_processed.json'

# Load the JSON data
with open(json_file, 'r') as f:
    data_json = json.load(f)


filtered_data = np.array(data_json["EEG"]["filtered_data"])

# Print the shape of the filtered data for verification
print("Shape of filtered EEG data:", filtered_data.shape)

# Compute the numerical rank of the filtered data.
# This rank reflects the number of linearly independent columns (channels).
data_rank = np.linalg.matrix_rank(filtered_data)
print("Effective rank of the filtered EEG data:", data_rank)