import os
import librosa
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


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
kfold = KFold(n_splits=20, shuffle=True, random_state=42)
results = []

for frame_length in frame_lengths:
    print(f"\nProcessing frame length {frame_length} s:")
    features = [extract_mfcc(path, frame_length=frame_length) for path in paths]
    fold_acc = []

    for train_index, test_index in kfold.split(features, encoded_labels):
        X_train, X_test = [features[i] for i in train_index], [features[i] for i in test_index]
        y_train, y_test = [encoded_labels[i] for i in train_index], [encoded_labels[i] for i in test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = SVC(degree=10, kernel='poly')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        fold_acc.append(acc)

        print(f"Fold {len(fold_acc)} accuracy: {acc:.4f}")

    mean_acc = np.mean(fold_acc)
    std_acc = np.std(fold_acc)
    results.append({
        "frame_length": frame_length,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc
    })
    print(f"Frame length {frame_length} s - Mean Accuracy: {mean_acc:.4f}, StdDev: {std_acc:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv("/home/workspace/paper-verif/cross-valid-res.csv", index=False)
