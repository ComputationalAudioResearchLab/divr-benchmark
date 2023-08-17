import pandas as pd
import json
import os
import matplotlib.pyplot as plt

# Load the data
with open('/home/workspace/paper-verif/preprocessed/svd.json') as f:
    data = json.load(f)


# extract diagnosis names
def get_diagnosis_names(diagnosis):
    if diagnosis['parent'] is not None:
        return [diagnosis['name']] + get_diagnosis_names(diagnosis['parent'])
    else:
        return [diagnosis['name']]


# Flatten the data and store it in a list
flattened_data = []
for item in data:
    for diagnosis in item['diagnosis']:
        diagnosis_names = get_diagnosis_names(diagnosis)
        for file in item['files']:
            for diagnosis_name in diagnosis_names:
                flattened_data.append({
                    'id': item['id'],
                    'gender': item['gender'],
                    'diagnosis': diagnosis_name,
                    'file': os.path.basename(file['path']),
                    'file_contains_a_h': 'a_h' in os.path.basename(file['path']),
                    'file_is_phrase': 'phrase' in os.path.basename(file['path'])
                })

df = pd.DataFrame(flattened_data)
print(len(df))
labels_of_interest = ['healthy', 'pathological', 'hyperfunktionelle dysphonie',
                      'laryngitis', 'hypofunktionelle dysphonie']

# filter the dataframe
df = df[df['diagnosis'].isin(labels_of_interest)]

# 2 tasks
# df_a_h = df[df['file_contains_a_h'] == True]
# df_phrase = df[df['file_is_phrase'] == True]

df_a_h = df[df['file_contains_a_h']]
df_phrase = df[df['file_is_phrase']]
print(len(df_a_h))
print(len(df_phrase))

def plot_graph(df, title):
    # Replace diagnosis names with newline characters for plotting
    df = df.replace({
        'healthy':'Healthy\nSpeech',
        'pathological':'Pathological\nSpeech',
        'hyperfunktionelle dysphonie': 'hyperkinetic\nDysphonia',
        'laryngitis': 'Reflux\nLaryngitis',
        'hypofunktionelle dysphonie': 'hypokinetic\nDysphonia',
    })

    # Count the occurrences of each diagnosis by gender
    gender_diagnosis_counts = df.groupby(['diagnosis', 'gender']).size().unstack()

    # Add a 'both' column
    gender_diagnosis_counts['both'] = gender_diagnosis_counts['male'] + gender_diagnosis_counts['female']

    # Calculate the sum of the three diagnoses for each gender
    pathological_counts = gender_diagnosis_counts.loc[['hyperkinetic\nDysphonia', 'Reflux\nLaryngitis', 'hypokinetic\nDysphonia']].sum()
    gender_diagnosis_counts.loc['Pathological\nSpeech'] = pathological_counts

    # Arrange the diagnoses in the specified order
    gender_diagnosis_counts = gender_diagnosis_counts.reindex(['Healthy\nSpeech', 'Pathological\nSpeech', 'hyperkinetic\nDysphonia', 'Reflux\nLaryngitis', 'hypokinetic\nDysphonia'])
    gender_diagnosis_counts = gender_diagnosis_counts[['male', 'female', 'both']]

    colors = ['blue', 'orange', 'green']

    ax = gender_diagnosis_counts.plot(kind='bar', color=colors)
    plt.xlabel('Diagnosis')
    plt.ylabel('Number of Patients')
    plt.title(title)
    plt.xticks(rotation=0)  # Make x labels horizontal

    # Add numbers on top of each bar
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005 - 0.01, p.get_height() * 1.005), fontsize=7)

    plt.show()

    plt.close()



# Plot the two graphs
plot_graph(df_a_h, 'vowel pronunciation task (for the vowel /a/)')
plot_graph(df_phrase, 'sentence pronunciation task')