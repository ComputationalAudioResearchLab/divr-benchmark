import os
import librosa
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Preprocessing function
def preprocess_audio(audio_path, target_sampling_rate=16000, top_db=1):
    signal, sr = librosa.load(audio_path, sr=target_sampling_rate)

    # trim silence
    signal_trimmed, _ = librosa.effects.trim(signal, top_db=top_db)

    return signal_trimmed

# Adjust n_fft based on signal length to get rid of warnings
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

# Extract MFCC features
frame_length = 0.025  # Just use this frame length
print(f"\nProcessing frame length {frame_length} s:")

features = [extract_mfcc(path, frame_length=frame_length) for path in paths]

# ...
# Convert the list of features to a numpy array
features_np = np.array(features)

# Now, we'll use t-SNE to reduce the dimensionality of the MFCC features to 2D
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features_np)
# ...


# Now, let's plot the 2D features. We'll use different colors for different classes.
plt.figure(figsize=(10, 10))

for label in np.unique(labels):
    # Find rows belonging to this label
    indices = [i for i, x in enumerate(labels) if x == label]
    subset = features_2d[indices]
    plt.scatter(subset[:, 0], subset[:, 1], label=label)

plt.legend()
plt.show()

# PCM