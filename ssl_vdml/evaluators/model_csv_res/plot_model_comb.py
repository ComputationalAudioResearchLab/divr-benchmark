import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the CSV file
df = pd.read_csv('/home/workspace/ssl_vdml/evaluators/model_csv_res/res.csv')

# Extract model and layer information from the log_file column
def extract_model_and_layer(log_file):
    model_match = re.search(r'/data/(\w+)\[', log_file)
    if not model_match:
        return "unknown"
    model_name = model_match.group(1)
    layers_match = re.findall(r'\[(\d+)\]', log_file)
    if not layers_match:
        return "unknown"
    label = f"{model_name}: {' & '.join(layers_match)}"
    return label

df['model_layer_label'] = df['log_file'].apply(extract_model_and_layer)

# Group by model_layer_label and compute the average accuracies
grouped_df = df.groupby('model_layer_label').agg({'train_acc': 'mean', 'val_acc': 'mean'}).reset_index()

# Sort the grouped data by validation accuracy from best to worst
sorted_grouped_df = grouped_df.sort_values(by='val_acc', ascending=False)

# Bar plot with sorted labels
fig, ax = plt.subplots(figsize=(14, 7))
bar_width = 0.35
index_sorted = np.arange(len(sorted_grouped_df['model_layer_label']))
bar1_sorted = ax.bar(index_sorted, sorted_grouped_df['train_acc'], bar_width, label='Train Acc', color='b', alpha=0.7)
bar2_sorted = ax.bar(index_sorted + bar_width, sorted_grouped_df['val_acc'], bar_width, label='Validation Acc', color='r', alpha=0.7)
for idx, rect in enumerate(bar1_sorted):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., height, '%.2f' % height, ha='center', va='bottom')
for idx, rect in enumerate(bar2_sorted):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., height, '%.2f' % height, ha='center', va='bottom')
ax.set_xlabel('Model & Layers')
ax.set_ylabel('Accuracy')
ax.set_title('Sorted Train and Validation Accuracies (Best to Worst)')
ax.set_xticks(index_sorted + bar_width / 2)
ax.set_xticklabels(sorted_grouped_df['model_layer_label'], rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.show()

# Prepare data for the violin plot
melted_df = df.melt(id_vars=['model_layer_label'], value_vars=['train_acc', 'val_acc'], 
                    var_name='Type', value_name='Accuracy')

# Filter data to only include validation accuracy
val_acc_df = melted_df[melted_df['Type'] == 'val_acc']

# Violin plot for validation accuracy
plt.figure(figsize=(14, 7))
sns.violinplot(x="model_layer_label", y="Accuracy", data=val_acc_df, inner="quartile", order=sorted_grouped_df['model_layer_label'])
plt.title('Violin Plot of Validation Accuracies')
plt.xlabel('Model & Layers')
plt.ylabel('Validation Accuracy')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
