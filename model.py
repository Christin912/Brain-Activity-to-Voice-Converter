# train_model.py

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, welch, iirnotch
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
import joblib  # For saving the scaler
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------
# Define Filters
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
# Load and Preprocess Data
# ---------------------------------------------

# Parameters
fs = 512  # Sampling frequency (Hz)
lowcut = 1.0  # Low cutoff frequency
highcut = 50.0  # High cutoff frequency
time_steps = 512  # Number of time steps per sample
num_channels = 1  # Single-channel data

# File paths (replace these with your actual file paths)
yes_files = [f"Samples/Yes/Data_yes{i}.npz" for i in range(1, 26)]
no_files = [f"Samples/No/Data_no{i}.npz" for i in range(1, 26)]

# Helper function to load and preprocess files
def load_and_preprocess(files, time_steps):
    """
    Loads EEG data from .npz files, splits into segments, and preprocesses.
    """
    data = []
    for file in files:
        try:
            with np.load(file) as npz_file:
                eeg_data = npz_file['values']
                total_samples = len(eeg_data)
                num_segments = total_samples // time_steps
                for i in range(num_segments):
                    segment = eeg_data[i*time_steps:(i+1)*time_steps]
                    
                    # Artifact removal: Notch filter at 50 Hz
                    segment = notch_filter(segment, fs)
                    
                    # Bandpass filter between 1-50 Hz
                    segment = bandpass_filter(segment, lowcut, highcut, fs)
                    data.append(segment)
        except KeyError:
            print(f"Key 'values' not found in {file}. Skipping.")
        except FileNotFoundError:
            print(f"File {file} not found. Skipping.")
    return np.array(data)  # Shape: [num_samples, time_steps]

# Load and preprocess data
yes_data = load_and_preprocess(yes_files, time_steps)
no_data = load_and_preprocess(no_files, time_steps)

print(f"Number of 'yes' samples: {yes_data.shape[0]}")
print(f"Number of 'no' samples: {no_data.shape[0]}")

# Create labels
yes_labels = np.ones(len(yes_data))  # Label 1 for "yes"
no_labels = np.zeros(len(no_data))   # Label 0 for "no"

# Combine 'yes' and 'no' data
X = np.concatenate((yes_data, no_data), axis=0)  # Shape: [num_samples, time_steps]
y = np.concatenate((yes_labels, no_labels), axis=0)  # Shape: [num_samples,]

print(f"Combined X shape: {X.shape}")
print(f"Combined y shape: {y.shape}")

# ---------------------------------------------
# Feature Engineering
# ---------------------------------------------

def extract_features(data, fs):
    """
    Extracts features from EEG data.
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

# Extract features
X_features = extract_features(X, fs)
print(f"Extracted features shape: {X_features.shape}")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# Save the scaler to a file
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved to 'scaler.pkl'")

# ---------------------------------------------
# Convert Data to PyTorch Tensors
# ---------------------------------------------

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Shape: [num_samples, 1]

# ---------------------------------------------
# Define the Neural Network
# ---------------------------------------------

class EEGNet(nn.Module):
    def __init__(self, input_size):
        super(EEGNet, self).__init__()
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
# Cross-Validation and Model Training
# ---------------------------------------------

# Define cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1

# Initialize lists to store metrics
all_accuracies = []
all_precisions = []
all_recalls = []
all_f1s = []
all_aucs = []

for train_index, test_index in kf.split(X_tensor, y_tensor):
    X_train_fold, X_test_fold = X_tensor[train_index], X_tensor[test_index]
    y_train_fold, y_test_fold = y_tensor[train_index], y_tensor[test_index]

    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train_fold, y_train_fold)
    test_dataset = torch.utils.data.TensorDataset(X_test_fold, y_test_fold)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize model, loss function, and optimizer
    input_size = X_train_fold.shape[1]
    model = EEGNet(input_size=input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    num_epochs = 100
    best_auc = 0
    patience = 10
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Evaluation on validation set
        model.eval()
        y_true_val = []
        y_prob_val = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(X_batch)
                y_true_val.extend(y_batch.cpu().numpy())
                y_prob_val.extend(y_pred.cpu().numpy())

        y_true_val = np.array(y_true_val)
        y_prob_val = np.array(y_prob_val).flatten()
        auc_val = roc_auc_score(y_true_val, y_prob_val)

        # Early stopping and saving the best model
        if auc_val > best_auc:
            best_auc = auc_val
            counter = 0
            best_model_state = model.state_dict()
            # Save the best model for the current fold
            model_filename = f'eegnet_fold{fold}_best.pth'
            torch.save(best_model_state, model_filename)
            print(f"Best model for fold {fold} saved at epoch {epoch + 1} as '{model_filename}'")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Load the best model
    model.load_state_dict(torch.load(f'eegnet_fold{fold}_best.pth'))

    # Evaluation on test set
    model.eval()
    y_true_list = []
    y_pred_list = []
    y_prob_list = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch)
            y_pred_binary = (y_pred > 0.5).float()
            y_true_list.extend(y_batch.cpu().numpy())
            y_pred_list.extend(y_pred_binary.cpu().numpy())
            y_prob_list.extend(y_pred.cpu().numpy())

    y_true_array = np.array(y_true_list)
    y_pred_array = np.array(y_pred_list)
    y_prob_array = np.array(y_prob_list).flatten()

    # Calculate metrics
    accuracy = accuracy_score(y_true_array, y_pred_array)
    precision = precision_score(y_true_array, y_pred_array, zero_division=0)
    recall = recall_score(y_true_array, y_pred_array, zero_division=0)
    f1 = f1_score(y_true_array, y_pred_array, zero_division=0)
    auc = roc_auc_score(y_true_array, y_prob_array)

    # Append metrics
    all_accuracies.append(accuracy)
    all_precisions.append(precision)
    all_recalls.append(recall)
    all_f1s.append(f1)
    all_aucs.append(auc)

    print(f"Fold {fold} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}\n")

    fold += 1

# Calculate mean metrics
mean_accuracy = np.mean(all_accuracies)
mean_precision = np.mean(all_precisions)
mean_recall = np.mean(all_recalls)
mean_f1 = np.mean(all_f1s)
mean_auc = np.mean(all_aucs)

print(f"Mean Accuracy across folds: {mean_accuracy:.4f}")
print(f"Mean Precision: {mean_precision:.4f}")
print(f"Mean Recall: {mean_recall:.4f}")
print(f"Mean F1 Score: {mean_f1:.4f}")
print(f"Mean AUC: {mean_auc:.4f}")

# ---------------------------------------------
# Optional: Train Final Model on Entire Dataset
# ---------------------------------------------

# Uncomment the following section if you wish to train on the entire dataset

# Step 7: Train Final Model on Entire Dataset
input_size = X_tensor.shape[1]
final_model = EEGNet(input_size=input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(final_model.parameters(), lr=0.001)

# Move model to device
final_model.to(device)

# Create DataLoader for the entire dataset
full_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=16, shuffle=True)

# Training loop for the final model
num_epochs_final = 100
best_auc_final = 0
patience_final = 10
counter_final = 0
best_model_state_final = None

for epoch in range(num_epochs_final):
    final_model.train()
    epoch_loss = 0
    for X_batch, y_batch in full_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        y_pred = final_model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Evaluation on the entire dataset
    final_model.eval()
    y_true_full = []
    y_prob_full = []

    with torch.no_grad():
        for X_batch, y_batch in full_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = final_model(X_batch)
            y_true_full.extend(y_batch.cpu().numpy())
            y_prob_full.extend(y_pred.cpu().numpy())

    y_true_full = np.array(y_true_full)
    y_prob_full = np.array(y_prob_full).flatten()
    auc_val_final = roc_auc_score(y_true_full, y_prob_full)

    # Early stopping and saving the best final model
    if auc_val_final > best_auc_final:
        best_auc_final = auc_val_final
        counter_final = 0
        best_model_state_final = final_model.state_dict()
        torch.save(best_model_state_final, 'eegnet_final_best.pth')
        print(f"Best final model saved at epoch {epoch + 1} as 'eegnet_final_best.pth'")
    else:
        counter_final += 1
        if counter_final >= patience_final:
            print(f"Early stopping for final model at epoch {epoch + 1}")
            break

    # Optionally print loss and AUC every few epochs
    if (epoch + 1) % 10 == 0 or epoch == 0:
        avg_loss = epoch_loss / len(full_loader)
        print(f"Final Model Epoch {epoch + 1}/{num_epochs_final}, Loss: {avg_loss:.4f}, Val AUC: {auc_val_final:.4f}")

# Load and save the best final model
final_model.load_state_dict(best_model_state_final)
torch.save(final_model.state_dict(), 'eegnet_final_best.pth')
print("Final model saved to 'eegnet_final_best.pth'")
