import os
import librosa
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def preprocess_audio(audio_path, target_sampling_rate=16000, top_db=1):
    signal, sr = librosa.load(audio_path, sr=target_sampling_rate)
    signal_trimmed, _ = librosa.effects.trim(signal, top_db=top_db)
    return signal_trimmed

def extract_mfcc(audio_path, frame_length=0.025, frame_stride=0.005, num_ceps=13):
    signal = preprocess_audio(audio_path)
    n_fft = min(int(frame_length*16000), len(signal))
    mfccs = librosa.feature.mfcc(y=signal, sr=16000, n_mfcc=num_ceps,
                                 hop_length=int(frame_stride*16000),
                                 n_fft=n_fft)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

def get_diagnosis_names(diagnosis):
    if diagnosis['parent'] is not None:
        return [diagnosis['name']] + get_diagnosis_names(diagnosis['parent'])
    else:
        return [diagnosis['name']]

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

frame_lengths = [0.025, 0.05, 0.1, 0.2, 0.5]
degrees = [5, 10, 15, 20]
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

results = []

for frame_length in frame_lengths:
    features = [extract_mfcc(path, frame_length=frame_length) for path in paths]
    X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)

    for degree in degrees:
        for kernel in kernels:
            model = SVC(degree=degree, kernel=kernel)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            results.append({
                'frame_length': frame_length,
                'degree': degree,
                'kernel': kernel,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            print(f"Frame length: {frame_length}, Degree: {degree}, Kernel: {kernel}, Accuracy: {accuracy:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv("/home/workspace/paper-verif/grid-search.csv", index=False)
