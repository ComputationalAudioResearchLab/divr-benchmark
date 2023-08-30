import os
import json
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim

# Preprocess the audio
def preprocess_audio(audio_path, target_sampling_rate=16000, top_db=1):
    signal, sr = librosa.load(audio_path, sr=target_sampling_rate)
    signal_trimmed, _ = librosa.effects.trim(signal, top_db=top_db)
    return signal_trimmed

# Extract MFCC features
def extract_mfcc(audio_path, frame_length=0.025, frame_stride=0.005, num_ceps=13):
    signal = preprocess_audio(audio_path)
    n_fft = min(int(frame_length*16000), len(signal))
    mfccs = librosa.feature.mfcc(y=signal, sr=16000, n_mfcc=num_ceps,
                                 hop_length=int(frame_stride*16000),
                                 n_fft=n_fft)
    return mfccs.T  # Transpose to match PyTorch expected input shape

# CNN-RNN Hybrid Model using PyTorch
class HybridModel(nn.Module):
    def __init__(self, input_shape):
        super(HybridModel, self).__init__()
        
        # Convolutional Blocks
        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv1d(input_shape if i == 0 else 64, 64, 3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ) for i in range(5)]
        )
        
        # RNN Block
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        
        # Dense Layers
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 2)  # Assuming 2 classes: Healthy and Pathological
    
    def forward(self, x):
        x = self.conv_layers(x)
        x, _ = self.lstm(x)
        x = self.fc1(x[:, -1, :])  # Take the last output of LSTM
        x = self.fc2(x)
        return x

# Load the data
with open('/home/workspace/paper-verif/preprocessed/svd.json') as f:
    data = json.load(f)

# Specify the labels of interest and populate DataFrame
labels_of_interest = ['healthy', 'pathological']
flattened_data = []
for item in data:
    for diagnosis in item['diagnosis']:
        diagnosis_name = diagnosis['name']
        if diagnosis_name in labels_of_interest:
            for file in item['files']:
                if 'a_n' in os.path.basename(file['path']):
                    flattened_data.append({
                        'id': item['id'],
                        'gender': item['gender'],
                        'diagnosis': diagnosis_name,
                        'file': os.path.basename(file['path']),
                        'file_path': file['path']
                    })

df = pd.DataFrame(flattened_data)

# Prepare features and labels
paths = df['file_path'].tolist()
labels = df['diagnosis'].tolist()
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)

# Extract features
features = [extract_mfcc(path) for path in paths]

# Prepare data for PyTorch (convert to tensors, etc.)
X = torch.tensor(features, dtype=torch.float32)
y = torch.tensor(encoded_labels, dtype=torch.long)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
input_shape = X_train.shape[1]
model = HybridModel(input_shape)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):  # Number of epochs
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    _, predicted = torch.max(y_pred, 1)
    report = classification_report(y_test, predicted, target_names=le.classes_[np.unique(y_test)], zero_division=0)
    print("Evaluating model...")
    print(report)
