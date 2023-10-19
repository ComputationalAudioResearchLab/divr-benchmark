import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# File paths for all models
file_paths = {
    'data2vec': '/home/workspace/ssl_vdml/evaluators/csv_results/data2vec_crunched_accuracies.csv',
    'decoar2': '/home/workspace/ssl_vdml/evaluators/csv_results/decoar2_crunched_accuracies.csv',
    'hubert': '/home/workspace/ssl_vdml/evaluators/csv_results/hubert_crunched_accuracies.csv',
    'wav2vec_large': '/home/workspace/ssl_vdml/evaluators/csv_results/wav2vec_large_crunched_accuracies.csv',
    'wavlm_large': '/home/workspace/ssl_vdml/evaluators/csv_results/wavlm_large_crunched_accuracies.csv'
}

# Extract layer information from the log_file column
def extract_layer_v2(log_file):
    if "full" in log_file:
        return "full"
    match = re.search(r'_latents\[(\d+)\]', log_file)
    if match:
        return match.group(1)
    return "unknown"

# Read and process all CSV files
dfs = {}
for model, path in file_paths.items():
    df = pd.read_csv(path)
    df['layer'] = df['log_file'].apply(extract_layer_v2)
    dfs[model] = df

# Bar plot function
def plot_bar(df, ax, title):
    grouped = df.groupby('layer').agg({'train_acc': 'mean', 'val_acc': 'mean'}).reset_index()
    sorted_grouped = grouped.sort_values(by='val_acc', ascending=False)
    index_sorted = np.arange(len(sorted_grouped['layer'])) 
    bar1_sorted = ax.bar(index_sorted, sorted_grouped['train_acc'], bar_width, label='Train Acc', color='b', alpha=0.7)
    bar2_sorted = ax.bar(index_sorted + bar_width, sorted_grouped['val_acc'], bar_width, label='Validation Acc', color='r', alpha=0.7)
    for idx, rect in enumerate(bar1_sorted):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height, '%.2f' % height, ha='center', va='bottom', fontsize=8)
    for idx, rect in enumerate(bar2_sorted):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height, '%.2f' % height, ha='center', va='bottom', fontsize=8)
    ax.set_title(title)
    ax.set_xticks(index_sorted + bar_width / 2)
    ax.set_xticklabels(sorted_grouped['layer'], rotation=90, fontsize=8)

# Violin plot function
def plot_violin(df, ax, title):
    melted_df = df.melt(id_vars=['layer'], value_vars=['val_acc'], var_name='Type', value_name='Accuracy')
    val_acc_df = melted_df[melted_df['Type'] == 'val_acc']
    val_acc_df['layer'] = pd.Categorical(val_acc_df['layer'], categories=sort_order, ordered=True)
    sns.violinplot(x="layer", y="Accuracy", data=val_acc_df, inner="quartile", order=sort_order, ax=ax)
    ax.set_title(title)
    ax.set_xticklabels(sort_order, rotation=90, fontsize=8)

# Bar plots for all models in subplots with a denser layout
bar_width = 0.35
sort_order = [str(i) for i in range(13)] + ["full"]
fig, axes = plt.subplots(5, 1, figsize=(10, 10))
for i, (model, df) in enumerate(dfs.items()):
    plot_bar(df, axes[i], model)
    axes[i].set_title(model, fontsize=10)  # Adjust title font size
axes[4].set_xlabel('Layer', fontsize=12)
axes[2].set_ylabel('Accuracy', fontsize=10)
axes[0].legend(loc="upper right", fontsize=8)
plt.tight_layout(pad=0.5)
plt.show()

# Violin plots for all models in subplots with a denser layout
fig, axes = plt.subplots(5, 1, figsize=(10, 10))
for i, (model, df) in enumerate(dfs.items()):
    plot_violin(df, axes[i], model)
    axes[i].set_title(model, fontsize=10)  # Adjust title font size
axes[4].set_xlabel('Layer', fontsize=12)
axes[2].set_ylabel('Validation Accuracy', fontsize=10)
plt.tight_layout(pad=0.5)
plt.show()
plt.close()
