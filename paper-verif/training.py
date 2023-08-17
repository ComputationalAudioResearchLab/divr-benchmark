import os
import librosa
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Preprocessing function
def preprocess_audio(audio_path, target_sampling_rate=16000, top_db=1):
    signal, sr = librosa.load(audio_path, sr=target_sampling_rate)

    # trim silence
    signal_trimmed, _ = librosa.effects.trim(signal, top_db=top_db)

    return signal_trimmed
# TODO: fix the nfft 1024
# adjust n_fft based on signal length to get rid of warnings
def extract_mfcc(audio_path, frame_length=0.025, frame_stride=0.005, num_ceps=13):
    signal = preprocess_audio(audio_path)
    n_fft = min(int(frame_length*16000), len(signal))  # Adjust n_fft based on signal length 
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

# Load the data
with open('/home/workspace/paper-verif/preprocessed/svd.json') as f:
    data = json.load(f)

# Specify the labels of interest
labels_of_interest = ['healthy', 'pathological', 'hyperfunktionelle dysphonie',
                      'laryngitis', 'hypofunktionelle dysphonie']

# Flatten the data and store it in a list
flattened_data = []
for item in data:
    for diagnosis in item['diagnosis']:
        diagnosis_names = get_diagnosis_names(diagnosis)
        for diagnosis_name in diagnosis_names:
            if diagnosis_name in labels_of_interest:
                for file in item['files']:
                    if 'a_n' in os.path.basename(file['path']):
                        if diagnosis_name in ['hyperfunktionelle dysphonie', 'laryngitis', 'hypofunktionelle dysphonie']:
                            diagnosis_name = 'pathological'  # Treat these diseases as 'pathological'
                        flattened_data.append({
                            'id': item['id'],
                            'gender': item['gender'],
                            'diagnosis': diagnosis_name,
                            'file': os.path.basename(file['path']),
                            'file_path': file['path']
                        })

df = pd.DataFrame(flattened_data)

# Collect all audio paths and corresponding labels
paths = df['file_path'].tolist()
labels = df['diagnosis'].tolist()

# labels
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)

frame_lengths = [0.025, 0.05, 0.1, 0.2, 0.5]  # Different frame lengths to use

for frame_length in frame_lengths:
    print(f"\nProcessing frame length {frame_length} s:")

    features = [extract_mfcc(path, frame_length=frame_length) for path in paths]

    print("Splitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
# multi processing doing grid search
# permute, itertools.permutation
# output into a sheet(csv)
    model = SVC(degree=5, kernel = 'poly')

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Evaluating model...")
    print(classification_report(y_test, y_pred, target_names=le.classes_[np.unique(y_test)], zero_division=0))
    
    # TODO:
    # Cross validation DONE
    # Grid search DONE
    # multi processing (speed up the process)
    # play with the mfcc config
    # MOVE everything on branch
    
    
    
