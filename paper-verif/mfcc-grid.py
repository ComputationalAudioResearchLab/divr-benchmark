import os
import librosa
import numpy as np
import pandas as pd
import json
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Preprocessing function
def preprocess_audio(audio_path, target_sampling_rate=16000, top_db=1):
    signal, sr = librosa.load(audio_path, sr=target_sampling_rate)
    signal_trimmed, _ = librosa.effects.trim(signal, top_db=top_db)
    return signal_trimmed

# Extract MFCC with more parameters
def extract_mfcc_v2(audio_path, frame_length=0.025, frame_stride=0.005, num_ceps=13, window='hann'):
    signal = preprocess_audio(audio_path)
    n_fft = min(int(frame_length*16000), len(signal))  # Adjust n_fft based on signal length
    hop_length = int(frame_stride*16000)
    mfccs = librosa.feature.mfcc(y=signal, sr=16000, n_mfcc=num_ceps,
                                 hop_length=hop_length, n_fft=n_fft, window=window)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

def get_diagnosis_names(diagnosis):
    if diagnosis['parent'] is not None:
        return [diagnosis['name']] + get_diagnosis_names(diagnosis['parent'])
    else:
        return [diagnosis['name']]

# Load the data
with open('/home/workspace/paper-verif/preprocessed/svd.json') as f:
    data = json.load(f)

labels_of_interest = ['healthy', 'pathological', 'hyperfunktionelle dysphonie',
                      'laryngitis', 'hypofunktionelle dysphonie']

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
paths = df['file_path'].tolist()
labels = df['diagnosis'].tolist()

le = LabelEncoder()
encoded_labels = le.fit_transform(labels)

# Set up a grid of values for the window parameter
windows = ['hann', 'hamming', 'blackman', 'bartlett', 'boxcar']

results = []

# Iterate over the grid and extract features for each window type
for window in windows:
    print(f"Processing parameters: window={window}")

    features = [extract_mfcc_v2(path, window=window) for path in paths]

    X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)
    
    model = SVC(degree=5, kernel='poly')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=le.classes_[np.unique(y_test)], 
                                   zero_division=0, output_dict=True)

    accuracy = report['accuracy']
    results.append({
        'window': window,
        'accuracy': accuracy
    })

results_df = pd.DataFrame(results)
results_df.to_csv('/home/workspace/paper-verif/mfcc-grid.csv', index=False)
