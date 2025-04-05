
import matplotlib.pyplot as plt
from data import EEGSequenceDataset  

folder = "data"               
marker_file = "P1_AllLifts.json"  
train_dataset = EEGSequenceDataset(sequences=..., normalize=True, 
                                   channel_means=..., channel_stds=...)
sample_idx = 0
past, future = train_dataset[sample_idx]

past_np = past.numpy()
future_np = future.numpy()

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