import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV
df = pd.read_csv("/home/workspace/paper-verif/grid-search.csv")

# Unique kernels
kernels = df['kernel'].unique()

# Plotting Accuracy vs. Frame Length for each Kernel
for kernel in kernels:
    subset = df[df['kernel'] == kernel]
    plt.figure()
    plt.plot(subset['frame_length'], subset['accuracy'], marker='o', label='Accuracy')
    plt.title(f'Accuracy vs. Frame Length ({kernel} Kernel)')
    plt.xlabel('Frame Length')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'accuracy_vs_frame_length_{kernel}.png')
    plt.close()

# Plotting Precision, Recall, and F1-Score vs. Frame Length for each Kernel
for kernel in kernels:
    subset = df[df['kernel'] == kernel]
    plt.figure()
    plt.plot(subset['frame_length'], subset['precision'], marker='o', label='Precision')
    plt.plot(subset['frame_length'], subset['recall'], marker='x', label='Recall')
    plt.plot(subset['frame_length'], subset['f1_score'], marker='s', label='F1-Score')
    plt.title(f'Metrics vs. Frame Length ({kernel} Kernel)')
    plt.xlabel('Frame Length')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'metrics_vs_frame_length_{kernel}.png')
    plt.close()
