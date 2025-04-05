import json
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Applies a Butterworth bandpass filter to the input data.
    
    Parameters:
        data (np.ndarray): 2D array (samples x channels).
        lowcut (float): Low cutoff frequency (Hz).
        highcut (float): High cutoff frequency (Hz).
        fs (float): Sampling frequency (Hz).
        order (int): Filter order.
        
    Returns:
        np.ndarray: Filtered data.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

def apply_ica(eeg_data, n_components=None, random_state=42):
    """
    Applies ICA to the EEG data using FastICA.
    
    Parameters:
        eeg_data (np.ndarray): 2D array (samples x channels).
        n_components (int or None): Number of ICA components to compute. If None,
                                    all channels are used.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        S (np.ndarray): Independent components (sources).
        A (np.ndarray): Estimated mixing matrix.
        reconstructed (np.ndarray): Reconstructed EEG data from the ICA components.
    """
    ica = FastICA(n_components=n_components, random_state=random_state)
    S = ica.fit_transform(eeg_data)  # Compute independent components (sources)
    A = ica.mixing_                  # Get the mixing matrix
    reconstructed = ica.inverse_transform(S)
    return S, A, reconstructed

def preprocess_eeg_with_ica(json_filepath, output_filepath=None):
    """
    Loads a JSON file with EEG, EMG, and KIN data, applies a 0.5–40Hz bandpass filter to the EEG data,
    selects only the allowed channels, and then runs ICA on the filtered EEG. The resulting JSON structure 
    is updated with the filtered data, ICA components, mixing matrix, and reconstructed EEG.
    
    Parameters:
        json_filepath (str): Path to the input JSON file.
        output_filepath (str, optional): If provided, writes the updated JSON structure to this file.
        
    Returns:
        dict: Updated JSON structure with additional EEG processing results.
    """
    # Load the JSON data
    with open(json_filepath, 'r') as f:
        data_json = json.load(f)
    
    # Extract EEG data and sampling rate
    eeg_data = np.array(data_json["EEG"]["data"])  # shape: (samples x channels)
    fs = data_json["EEG"]["sampling_rate"]
    
    # Keep only the allowed channels ---
    allowed_channels = [
        "F3", "Fz", "F4",
        "FC5", "FC1", "FC2", "FC6",
        "C3", "Cz", "C4",
        "CP5", "CP1", "CP2", "CP6",
    ]

    channel_names = data_json["EEG"]["names"]  
    indices = [i for i, name in enumerate(channel_names) if name in allowed_channels]
    eeg_data = eeg_data[:, indices]
    data_json["EEG"]["names"] = [channel_names[i] for i in indices]
    
    # Apply bandpass filter (0.5-40Hz) on the EEG data
    filtered_eeg = bandpass_filter(eeg_data, lowcut=0.5, highcut=40, fs=fs, order=5)
    
    # Apply ICA on the filtered EEG data
    # n_components=None uses all available channels.
    S, A, reconstructed = apply_ica(filtered_eeg, n_components=None)
    
    # Update JSON structure with filtered data and ICA results
    data_json["EEG"]["filtered_data"] = filtered_eeg.tolist()
    data_json["EEG"]["ica_components"] = S.tolist()        # Independent components
    data_json["EEG"]["ica_mixing_matrix"] = A.tolist()       # Mixing matrix
    data_json["EEG"]["reconstructed_data"] = reconstructed.tolist()
    
    # Optionally, save the processed data to an output file
    if output_filepath:
        with open(output_filepath, 'w') as f:
            json.dump(data_json, f, indent=4)
    
    return data_json

if __name__ == '__main__':
    input_json = 'ica code\HS_P1_S1.json'     
    output_json = f'HS_P1_S1_processed.json'
    processed_data = preprocess_eeg_with_ica(input_json, output_json)
    print("Processed EEG data with bandpass filtering and ICA saved to:", output_json)