import os
import json
import numpy as np
import pandas as pd
import librosa
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

# Initialize TensorBoard writer and time counter
writer = SummaryWriter('runs/experiment_1')
start_time = time.time()


def get_diagnosis_names(diagnosis):
    if diagnosis['parent'] is not None:
        return [diagnosis['name']] + get_diagnosis_names(diagnosis['parent'])
    else:
        return [diagnosis['name']]

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

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
        self.fc2 = nn.Linear(64, 2) # 2 classes
    
    def forward(self, x):
        x = self.conv_layers(x)  # Output from Conv layers
        x = x.view(x.size(0), -1, 64)  # Flatten the output for LSTM
        x, _ = self.lstm(x)  # Output from LSTM
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
        diagnosis_names = get_diagnosis_names(diagnosis)
        for diagnosis_name in diagnosis_names:
            if diagnosis_name in labels_of_interest:
                for file in item['files']:
                    if 'a_n' in os.path.basename(file['path']):
                        if diagnosis_name in ['hyperfunktionelle dysphonie', 'laryngitis', 'hypofunktionelle dysphonie']:
                            diagnosis_name = 'pathological'
                        flattened_data.append({
                            'id': item['id'],
                            'gender': item['gender'],
                            'diagnosis': diagnosis_name,
                            'file': os.path.basename(file['path']),
                            'file_path': file['path']
                        })

df = pd.DataFrame(flattened_data)

# Prepare features and labels
# TODO: store the length and double check
# MFCC diff too much bet unhealthy and healthy
# TODO: FIXME only have 1 label
path_length=[]
healthy_length=[]
# ... (previous code remains unchanged)

# Create a DataFrame to check the class distribution
df_counts = df['diagnosis'].value_counts().reset_index()
df_counts.columns = ['diagnosis', 'count']
print("Number of samples per class:")
print(df_counts)

# Undersampling the 'pathological' class
min_count = min(df_counts['count'])  # The count of the class with the least number of samples

# Create balanced DataFrame
df_healthy = df[df['diagnosis'] == 'healthy']
df_pathological = df[df['diagnosis'] == 'pathological'].sample(min_count, random_state=42)  # Randomly sample 'min_count' rows
df_balanced = pd.concat([df_healthy, df_pathological], axis=0)

# ... (rest of the code remains unchanged)

# Prepare features and labels
paths = df_balanced['file_path'].tolist()
labels = df_balanced['diagnosis'].tolist()


le = LabelEncoder()
encoded_labels = le.fit_transform(labels)

print(np.unique(labels))

features = [extract_mfcc(path) for path in paths]

for label, feature in zip(labels, features):
    if label == 'healthy':
        healthy_length.append(feature.shape[0])
    else:
        path_length.append(feature.shape[0])

print("Number of samples per class:")
print(df['diagnosis'].value_counts())
print("max path length:", max(path_length))
print("max healthy length:" , max(healthy_length))

# exit()

# find max seq
max_length = max([feature.shape[0] for feature in features])

# Pad sequences (double check the length)
features_padded = [np.pad(feature, ((0, max_length - feature.shape[0]), (0, 0)), 'constant') for feature in features]

features_np = np.stack(features_padded)

# rreshape the data (batch_size, num_channels, length)
features_np_reshaped = features_np.transpose(0, 2, 1)

X = torch.tensor(features_np_reshaped, dtype=torch.float32)

# Prepare 
y = torch.tensor(encoded_labels, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_shape = features_np_reshaped.shape[1]  # Number of MFCC features (channels)
model = HybridModel(input_shape)

# Calculate class weights
class_weights = 1. / torch.tensor([len(df_healthy), len(df_pathological)], dtype=torch.float32)
class_weights = class_weights / class_weights.sum()

# Initialize loss function with class weights
# criterion = nn.CrossEntropyLoss(weight=class_weights)

criterion = nn.CrossEntropyLoss()

# learning rate decreased as the loss is spikey need lower lr further more
# reduce lr 10 times , needs to train 10 times longer
# but depends on the loss graph
# TODO: get rif od redundant code
# TODO: 1e-5 to e7
# and check scalr domain
optimizer = optim.Adam(model.parameters(), lr=1e-7)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize epoch counter
epoch_counter = 0

# Training loop
for epoch in range(10000):  # You can change the number of epochs
    epoch_counter += 1
    for X_batch, y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Log loss to TensorBoard
    writer.add_scalar('Training loss', loss.item(), epoch)
    # TODO: every 10 epoch validate model and add validation loss
    # Don't add the internal param take too much time
    
    # Log histograms of model parameters every 10 epochs 
    if epoch_counter % 10 == 0:
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

    # Log metrics every 5 minutes
    current_time = time.time()
    if current_time - start_time >= 300:  # 300 seconds == 5 minutes
        # Log other metrics here if needed
        start_time = current_time
    
    # more floating point dun add format
    print(f'Epoch [{epoch+1}/10], Loss: {loss.item()}')


#  Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    y_pred = model(X_test)
    _, predicted = torch.max(y_pred, 1)
    all_preds.extend(predicted.cpu().numpy())
    all_labels.extend(y_test.cpu().numpy())

# Generate the classification report
report = classification_report(all_labels, all_preds, target_names=le.classes_)
print(report)

# Generate and save the confusion matrix
plot_confusion_matrix(all_labels, all_preds, le.classes_)
plt.savefig("/home/workspace/paper-verif/confusion_matrix_without_weight.png")

# TODO: plot tensor board
# variation of batch size and learning rate 1e-4 and so on 10 times per epoch
# checkpoint the model in between, printing everything couple of minutes
# use tensor board to get the result per 
# loss every time, images and other stuff maybe 10 epoch
# try diff thing and combinations 
# search for more paper

# branch the new branch
# run preprocessed again to gerenate all the files
# change path in nn as well as generate features
# only run data2vec first
# think about how to automate testing the layer


# separate aiu from a, combination layer, mnoe exp