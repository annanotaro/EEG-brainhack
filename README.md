# EEG-brainhack

Welcome to the Brainhack project **"I Know What You Will Do: Forecasting Motor Behaviour from EEG Time Series"**.  
This repository contains tools, scripts, and documentation aimed at developing a deep learning solution for forecasting EEG time series to predict motor behavior.

## Dataset

We use the publicly available **WAY-EEG-GAL-ML** dataset, which contains EEG recordings related to motor activity.

🔗 [Download the Dataset](https://gin.g-node.org/matteo-d-m/way-eeg-gal-ml)

EEG Neural Signal Extraction with ICA
This code processes an EEG dataset to isolate clean neural signals using ICA. Starting from preprocessed EEG data  we applied a bandpass filter (1–40 Hz), re-referenced to common average, and set a standard 10–20 montage. Using MNE and mne-icalabel, we performed ICA to decompose signals and classify components (brain, eye blink, muscle artifact). Only components labeled as "brain" were retained and saved as a NumPy array for further analysis.
