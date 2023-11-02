import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Read the CSV file
data2vec_df = pd.read_csv('/home/workspace/ssl_vdml/evaluators/csv_results/wavlm_large_crunched_accuracies.csv')

# Extract layer information from the log_file column
def extract_layer_v2(log_file):
    if "full" in log_file:
        return "full"
    match = re.search(r'_latents\[(\d+)\]', log_file)
    if match:
        return match.group(1)
    return "unknown"

data2vec_df['layer'] = data2vec_df['log_file'].apply(extract_layer_v2)

# Group by layer and average the accuracies
data2vec_grouped = data2vec_df.groupby('layer').agg({'train_acc': 'mean', 'val_acc': 'mean'}).reset_index()

# Sort layers for plotting based on validation accuracy from best to worst
sorted_data2vec_grouped = data2vec_grouped.sort_values(by='val_acc', ascending=False)

# Bar plot with sorted layers
fig, ax = plt.subplots(figsize=(14, 7))
bar_width = 0.35
index_sorted = np.arange(len(sorted_data2vec_grouped['layer']))
bar1_sorted = ax.bar(index_sorted, sorted_data2vec_grouped['train_acc'], bar_width, label='Train Acc', color='b', alpha=0.7)
bar2_sorted = ax.bar(index_sorted + bar_width, sorted_data2vec_grouped['val_acc'], bar_width, label='Validation Acc', color='r', alpha=0.7)
for idx, rect in enumerate(bar1_sorted):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., height, '%.2f' % height, ha='center', va='bottom')
for idx, rect in enumerate(bar2_sorted):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., height, '%.2f' % height, ha='center', va='bottom')
ax.set_xlabel('Layer')
ax.set_ylabel('Accuracy')
# TODO:
ax.set_title('Sorted Train and Validation Accuracies for wavlm_large Model (Best to Worst)')
ax.set_xticks(index_sorted + bar_width / 2)
ax.set_xticklabels(sorted_data2vec_grouped['layer'])
ax.legend()
plt.tight_layout()
plt.show()

# Prepare data for violin plot
melted_data2vec = data2vec_df.melt(id_vars=['layer'], value_vars=['train_acc', 'val_acc'], 
                                   var_name='Type', value_name='Accuracy')

# Filter data to only include validation accuracy
val_acc_data2vec = melted_data2vec[melted_data2vec['Type'] == 'val_acc']

# Adjust the category order to place the "full" layer at the end
sort_order = [str(i) for i in range(13)] + ["full"]
val_acc_data2vec['layer'] = pd.Categorical(val_acc_data2vec['layer'], categories=sort_order, ordered=True)

# Violin plot for validation accuracy with adjusted order
plt.figure(figsize=(14, 7))
sns.violinplot(x="layer", y="Accuracy", data=val_acc_data2vec, inner="quartile", order=sort_order)
# TODO
plt.title('Violin Plot of Validation Accuracies for wavlm_large Model')
plt.xlabel('Layer')
plt.ylabel('Validation Accuracy')
plt.tight_layout()
plt.show()
