import numpy as np
import json
import os
import re

def windows(folder, filename):
    """
    Function to extract EEG data windows based on marker events.
    
    Parameters:
    folder (str): Directory containing EEG files.
    filename (str): JSON file containing marker data.
    
    Returns:
    all_sequences (list): List of tuples containing past and future EEG data windows.
        """
    # Directories where your EEG and marker files are stored
    eeg_dir = folder  # Directory containing .npy EEG files
    markers_file = filename # File containing marker data

    # Load the marker file and extract columns and data
    with open(markers_file, 'r') as f:
        marker_data = json.load(f)
    columns = marker_data["columns"]
    data_rows = marker_data["data"]

    def col_idx(col_name):
        return columns.index(col_name)

    # Build a dictionary mapping run numbers to LEDOn sample indices.
    # LEDOn is provided in seconds; convert to samples assuming a sampling rate of 500 Hz.
    markers_by_run = {}
    for row in data_rows:
        run = int(row[col_idx("Run")])
        led_on_sec = row[col_idx("LEDOn")]
        if led_on_sec is None:
            continue
        led_on_sample = int(led_on_sec * 500)
        if run not in markers_by_run:
            markers_by_run[run] = []
        markers_by_run[run].append(led_on_sample)

    # This list will hold all (past, future) pairs from all EEG series
    all_sequences = []

    # Iterate through EEG files and extract run number using regex.
    for eeg_filename in os.listdir(eeg_dir):
        if eeg_filename.endswith('.npy'):
            eeg_path = os.path.join(eeg_dir, eeg_filename)
            
            # Use regex to extract the run number from the filename.
            m = re.search(r'_S(\d+)', eeg_filename)
            if m:
                run = int(m.group(1))
            else:
                print(f"Could not extract run number from file name {eeg_filename}.")
                continue
            
            # Load the EEG data; expected shape: [32, N_samples]
            eeg_data = np.load(eeg_path)
            
            # Process LEDOn events for this run
            if run not in markers_by_run:
                print(f"No markers found for run {run} in {markers_file}.")
                continue
            
            for t in markers_by_run[run]:
                # Ensure the window [t-1000, t+1500) is within the valid range
                if t - 1000 >= 0 and t + 1500 <= eeg_data.shape[1]:
                    past_window = eeg_data[:, t - 1000:t]      # 1000 samples (2 seconds before event)
                    future_window = eeg_data[:, t:t + 1500]      # 1500 samples (3 seconds after event)
                    all_sequences.append((past_window, future_window))
                else:
                    print(f"Skipping event at sample {t} in run {run}: window out of bounds.")
    return all_sequences

if __name__ == "__main__":    
    folder = "data"
    filename = "P1_AllLifts.json"
    all_sequences = windows(folder, filename)
    print(f"Collected {len(all_sequences)} (past, future) pairs from all EEG series.")
    print(f"Example sequence shape: {all_sequences[0][0].shape}, {all_sequences[0][1].shape}")

