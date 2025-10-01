# Nat-Hack-2024
# Voice your thoughts, a disability communication aid.
  14/11/2024
  Made by Khushdeep Brar, Angad Chahil, Annie Ding, Duy Bui Nguyen Khuong, Christin Anil, Amanpreet Sekhon

Brain Activity to Voice Converter

A prototype brain–computer interface (BCI) that translates EEG signals into reliable yes/no vocal responses. This project was developed to improve communication options for non-verbal users.

Features

Signal Processing Pipeline

Data acquisition via Arduino

Filtering and artifact removal

Time & frequency domain feature extraction

Machine Learning

Neural network models built with PyTorch

Cross-validation & class-weighting to handle imbalanced data

Achieved ~70% accuracy on held-out test data

Visualization & Analysis

Tools built with Matplotlib to interpret model behaviour

Assisted in feature engineering and iteration cycles

Tech Stack

Hardware: Arduino, EEG headset

Languages & Tools: Python, PyTorch, Matplotlib, NumPy, Scikit-learn

Methodologies: Neural networks, signal processing, cross-validation

Getting Started

Clone the repository

git clone https://github.com/Khushdeep1337/Nat-Hack-2024.git
cd Nat-Hack-2024


Install dependencies

pip install -r requirements.txt


Connect your EEG acquisition hardware (Arduino setup recommended).

Run preprocessing scripts to generate ML-ready datasets.

Train the model:

python train.py


Evaluate results and visualize outputs with:

python visualize.py
