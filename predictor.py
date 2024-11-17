# predictor2.py

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, welch, iirnotch
from scipy.stats import skew, kurtosis
import joblib  # For loading the scaler
import torch
import torch.nn as nn
import warnings

# Suppress FutureWarning from torch.load
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------
# Define Filters (Must Match Training)
# ---------------------------------------------

def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Creates a Butterworth bandpass filter.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def notch_filter(data, fs, freq=50.0, quality_factor=30.0):
    """
    Applies a notch filter to remove power line noise at 50 Hz.
    """
    nyquist = 0.5 * fs
    freq = freq / nyquist
    b, a = iirnotch(freq, quality_factor)
    return lfilter(b, a, data, axis=0)

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Applies a Butterworth bandpass filter to the data.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data, axis=0)

# ---------------------------------------------
# Load and Preprocess New Data
# ---------------------------------------------

# Parameters
fs = 512  # Sampling frequency (Hz)
lowcut = 1.0  # Low cutoff frequency
highcut = 50.0  # High cutoff frequency
time_steps = 512  # Number of time steps per sample

# Prompt user for the new data file path
new_file = input("Enter the path to the new EEG data file: ")

def load_new_data(file):
    """
    Loads new EEG data from a .npz file.
    """
    try:
        with np.load(file) as npz_file:
            return npz_file['values']
    except KeyError:
        print(f"Key 'values' not found in {file}.")
        return np.empty((0, time_steps))
    except FileNotFoundError:
        print(f"File {file} not found.")
        return np.empty((0, time_steps))

# Load new data
new_data = load_new_data(new_file)

if new_data.size == 0:
    print("No data to process.")
    exit()

print(f"Number of new samples: {len(new_data)}")

# Preprocess new data
def preprocess_new_data(data, fs, lowcut, highcut, time_steps):
    """
    Preprocesses new EEG data by applying filters and segmenting.
    """
    processed_data = []
    total_samples = len(data)
    num_segments = total_samples // time_steps
    for i in range(num_segments):
        segment = data[i*time_steps:(i+1)*time_steps]
        # Artifact removal: Notch filter at 50 Hz
        segment = notch_filter(segment, fs)
        # Bandpass filter between 1-50 Hz
        segment = bandpass_filter(segment, lowcut, highcut, fs)
        processed_data.append(segment)
    return np.array(processed_data)

# Apply preprocessing
processed_new_data = preprocess_new_data(new_data, fs, lowcut, highcut, time_steps)
print(f"Processed new data shape: {processed_new_data.shape}")  # [num_samples, time_steps]

# ---------------------------------------------
# Feature Engineering (Must Match Training)
# ---------------------------------------------

def extract_features_new(data, fs):
    """
    Extracts features from new EEG data.
    """
    features = []
    for sample in data:
        # Flatten sample if necessary
        sample = sample.flatten()

        # Power Spectral Density (PSD) features
        freqs, psd = welch(sample, fs=fs, nperseg=256)
        psd_mean = np.mean(psd)
        psd_std = np.std(psd)
        psd_skew = skew(psd)
        psd_kurt = kurtosis(psd)

        # Time-domain statistical features
        mean = np.mean(sample)
        std = np.std(sample)
        skewness = skew(sample)
        kurt = kurtosis(sample)

        # Combine all features
        sample_features = [
            psd_mean, psd_std, psd_skew, psd_kurt,
            mean, std, skewness, kurt
        ]
        features.append(sample_features)
    return np.array(features)

# Extract features from new data
X_new_features = extract_features_new(processed_new_data, fs)
print(f"Extracted features from new data shape: {X_new_features.shape}")  # [num_samples, 8]

# ---------------------------------------------
# Standardize Features (Using Saved Scaler)
# ---------------------------------------------

# Load the scaler
scaler = joblib.load('scaler.pkl')
print("Scaler loaded from 'scaler.pkl'")

# Standardize new features
X_new_scaled = scaler.transform(X_new_features)

# ---------------------------------------------
# Define the Neural Network (Must Match Training)
# ---------------------------------------------

class EEGNet(nn.Module):
    def __init__(self, input_size):
        super(EEGNet, self).__init__()
        # Ensure this architecture matches your training script
        self.fc1 = nn.Linear(input_size, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 8)
        self.bn2 = nn.BatchNorm1d(8)
        self.fc3 = nn.Linear(8, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

# ---------------------------------------------
# Load the Trained Model
# ---------------------------------------------

# Initialize model
input_size = X_new_scaled.shape[1]  # Should be 8
model = EEGNet(input_size=input_size)

# Load the saved model state
# Replace 'eegnet_final_best.pth' with your model file if different
model.load_state_dict(torch.load('eegnet_final_best.pth', map_location=torch.device('cpu')))
print("Model loaded from 'eegnet_final_best.pth'")

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ---------------------------------------------
# Make Predictions
# ---------------------------------------------

# Convert to PyTorch tensor
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    y_pred = model(X_new_tensor)
    y_pred_binary = (y_pred > 0.5).float()

# ---------------------------------------------
# Output Predictions
# ---------------------------------------------

# Calculate the number of positive predictions
num_positive = y_pred_binary.sum().item()
num_total = y_pred_binary.shape[0]

# Determine the final answer based on the majority
if num_positive / num_total >= 0.5:
    final_answer = "Yes"
else:
    final_answer = "No"

# Output the final answer
print(f"The final answer is: {final_answer}")