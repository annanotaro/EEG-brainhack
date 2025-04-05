import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sequences import windows

# Define the folder and filename for the EEG data
folder = "data"
filename = "P1_AllLifts.json"
all_sequences = windows(folder, filename)

# Split the data into training and testing sets (e.g., 80% train, 20% test)
np.random.shuffle(all_sequences)
split_idx = int(0.8 * len(all_sequences))
train_sequences = all_sequences[:split_idx]
test_sequences = all_sequences[split_idx:]

# Compute mean and standard deviation for each channel using only the training set
train_data_concat = np.concatenate(
    [np.concatenate(seq, axis=1) for seq in train_sequences],
    axis=1
)  # Result: [14, total_time_points]
# print (f"train_data_concat shape: {train_data_concat.shape}") = train_data_concat shape: (14, 587500)

channel_means = train_data_concat.mean(axis=1)
channel_stds = train_data_concat.std(axis=1)
channel_stds[channel_stds < 1e-6] = 1e-6  # Avoid division by zero

# Define a custom Dataset class for EEG sequences 
class EEGSequenceDataset(Dataset):
    def __init__(self, sequences, normalize=True, channel_means=None, channel_stds=None):
        self.sequences = sequences
        self.normalize = normalize
        if normalize:
            assert channel_means is not None and channel_stds is not None, "Mean and std must be provided for normalization."
            self.means = torch.tensor(channel_means, dtype=torch.float32).reshape(14, 1)
            self.stds = torch.tensor(channel_stds, dtype=torch.float32).reshape(14, 1)
        else:
            self.means = torch.zeros((14, 1), dtype=torch.float32)
            self.stds = torch.ones((14, 1), dtype=torch.float32)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Retrieve the past and future sequences for the given index
        past_np, future_np = self.sequences[idx]
        past = torch.from_numpy(past_np.astype(np.float32))
        future = torch.from_numpy(future_np.astype(np.float32))
        if self.normalize:
            # Normalize the data using the computed means and standard deviations
            past = (past - self.means) / self.stds
            future = (future - self.means) / self.stds
        return past, future

# Create DataLoaders for training and testing
train_dataset = EEGSequenceDataset(train_sequences, normalize=True, channel_means=channel_means, channel_stds=channel_stds)
test_dataset = EEGSequenceDataset(test_sequences, normalize=True, channel_means=channel_means, channel_stds=channel_stds)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


import matplotlib.pyplot as plt

# Retrieve a sample from the training dataset
sample_idx = 0  # For example, the first sample
past, future = train_dataset[sample_idx]

# Convert tensors to NumPy arrays for plotting
past_np = past.numpy()
future_np = future.numpy()

# Let's visualize the first channel (index 0) of the past and future windows
fig, ax = plt.subplots(2, 1, figsize=(12, 6))

# Plot the past window (2 seconds before the event)
ax[0].plot(past_np[0, :])
ax[0].set_title("Past Window (Channel 1) - 2 seconds before event")
ax[0].set_xlabel("Time (samples)")
ax[0].set_ylabel("Normalized Amplitude")

# Plot the future window (3 seconds after the event)
ax[1].plot(future_np[0, :])
ax[1].set_title("Future Window (Channel 1) - 3 seconds after event")
ax[1].set_xlabel("Time (samples)")
ax[1].set_ylabel("Normalized Amplitude")

plt.tight_layout()
plt.show()