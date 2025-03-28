import json
import numpy as np
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components


json_file = 'HS_P1_S1_processed.json'
with open(json_file, 'r') as f:
    data_json = json.load(f)

eeg_dict = data_json["EEG"]
filtered_data = np.array(eeg_dict["filtered_data"])
fs = eeg_dict["sampling_rate"]
provided_names = eeg_dict["names"]  # list of channel names from your JSON

# 2. Create an MNE Raw object
# MNE RawArray expects data in shape (n_channels, n_times), so we transpose.
raw_data = filtered_data.T  # now shape: (n_channels, n_times)
n_channels, n_times = raw_data.shape

# Create default channel names (e.g., "EEG1", "EEG2", ...) for initial info
default_names = [f"EEG{i+1}" for i in range(n_channels)]
info = mne.create_info(ch_names=default_names, sfreq=fs, ch_types=["eeg"] * n_channels)
raw = mne.io.RawArray(raw_data, info)
print("Raw object info (before renaming):")
print(raw.info)

# 3. Rename channels to match provided names
if len(provided_names) != n_channels:
    raise ValueError("The number of provided channel names does not match the number of channels in the data.")

rename_mapping = dict(zip(raw.info["ch_names"], provided_names))
raw.rename_channels(rename_mapping)
print("Raw object info (after renaming):")
print(raw.info)

# 4. Set Montage and Reference
# Set a standard 10-20 montage and a common average reference.
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)
raw.set_eeg_reference('average', projection=False)

# 5. Fit ICA explicitly on the Raw data
ica = ICA(n_components=31, random_state=97, max_iter='auto')
ica.fit(raw)
print("ICA fitted successfully.")

# 6. Apply ICLabel using mne-icalabel
labels_dict = label_components(raw, ica, method='iclabel')
print("ICLabel predicted labels:")
for i, (label, prob) in enumerate(zip(labels_dict["labels"], labels_dict["y_pred_proba"]), start=1):
    print(f"Label {i}: {label}, Prob: {int(100 * prob)}%")

# 7. Extract only neural components 
neural_indices = [i for i, lab in enumerate(labels_dict["labels"]) if lab == "brain"]
print("Indices of neural components (Brain):", neural_indices)

sources = ica.get_sources(raw).get_data()  # shape: (n_components, n_times)
neural_sources = sources[neural_indices, :]
print("Shape of neural sources:", neural_sources.shape)

np.save("HS_P1_S1_neural_components.npy", neural_sources)
print("Neural components saved to 'HS_P1_S1_neural_components.npy'")