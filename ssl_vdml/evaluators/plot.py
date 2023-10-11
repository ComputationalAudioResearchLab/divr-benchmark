import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data_path = '/home/workspace/ssl_vdml/evaluators/crunch_val_acc.py'  # Replace with your file path
data = pd.read_csv(data_path)

# Extract layer information
data['layer'] = data['log_file'].apply(lambda x: int(x.split('[')[1][0]))

# Bar Plot
layer_means = data.groupby('layer')[['train_acc', 'val_acc']].mean()

plt.figure(figsize=(10, 6))
plt.bar(layer_means.index.astype(str), layer_means['train_acc'], width=0.4, label='Training Accuracy')
plt.bar(layer_means.index.astype(str), layer_means['val_acc'], width=0.4, bottom=layer_means['train_acc'], label='Validation Accuracy')
plt.xlabel('Layer')
plt.ylabel('Accuracy')
plt.title('Comparison of Training and Validation Accuracy for Different Layers')
plt.legend()
plt.grid(axis='y')
plt.show()

# Violin Plot
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
sns.violinplot(x=data['layer'].astype(str), y=data['val_acc'], inner="quartile")
plt.title('Distribution of Validation Accuracy Across Different Layers')
plt.xlabel('Layer')
plt.ylabel('Validation Accuracy')
plt.show()
