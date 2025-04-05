# EEG-brainhack
Welcome to the Brainhack project "I Know What You Will Do: Forecasting Motor Behaviour from EEG Time Series".
This repository contains tools, scripts, and documentation aimed at developing a deep learning solution for forecasting EEG time series to predict motor behavior.

Dataset
We use the publicly available WAY-EEG-GAL-ML dataset, which contains EEG recordings related to motor activity.

🔗 Download the Dataset

EEG Neural Signal Extraction with ICA
Our code processes an EEG dataset to isolate clean neural signals using Independent Component Analysis (ICA). Starting from preprocessed EEG data, we applied a bandpass filter (0.5–40 Hz), re-referenced the signals to a common average, and set a standard 10–20 montage. Then, using MNE and mne-icalabel, we decomposed the EEG signals into independent components, classified them (e.g., brain, eye blink, muscle artifact), and retained only the components labeled as "brain". The neural components are then used to reconstruct a cleaned EEG signal, which is normalized and saved as a NumPy array for further analysis.

# Dataset Formatting for Event-Related Windows
Our pipeline further processes the cleaned EEG signals by extracting time windows around behavioral events. In particular, we focus on the LEDOn event (which signals the start of a motor task). For each trial, we extract:

Past window: 2 seconds before the LEDOn event (1000 samples at 500 Hz).

Future window: 3 seconds after the LEDOn event (1500 samples at 500 Hz).

The function windows() in sequences.py reads marker data from a JSON file (e.g., P1_AllLifts.json) and loads the corresponding EEG data (stored as .npy files). It uses the run number (extracted via a regular expression) to match markers with the correct EEG recording and then extracts valid (past, future) window pairs. In our case, each window pair has shapes (14, 1000) for the past and (14, 1500) for the future, since we work with 14 selected EEG channels.

Retained Channels
During preprocessing (see bandpass_filter.py), we retain only 14 specific channels from the original EEG data. These channels are:

Frontal: F3, Fz, F4
Frontal-Central: FC5, FC1, FC2, FC6
Central: C3, Cz, C4
Central-Parietal: CP5, CP1, CP2, CP6

This channel selection focuses on regions critical for motor behavior and ensures consistency in subsequent analysis.

# Data Preparation and Custom Dataset
The extracted window pairs are then split into training and testing sets. In data.py:

The training windows are concatenated along the time axis to form a single array (called train_data_concat) of shape (14, total_time_points), which is used to compute per-channel mean and standard deviation.

These statistics are used to normalize the EEG data.

A custom PyTorch Dataset class, EEGSequenceDataset, is defined to load each (past, future) pair and apply the normalization if enabled. 

DataLoaders are created to efficiently iterate over the dataset in batches during model training.