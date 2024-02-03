from pathlib import Path
from matplotlib import pyplot as plt
import yaml
import numpy as np


def load_labels(input_path: Path):
    with open(input_path, "r") as map_file:
        data = yaml.load(map_file, yaml.FullLoader)
        for idx, item in enumerate(data):
            item_votes = list(item["votes"].values())
            label_data = np.unique(item_votes, return_counts=True)
            label_values = list(label_data[0])
            label_counts = list(label_data[1])
            label_data = zip(label_values, label_counts)
            sorted_label_data = sorted(label_data, key=lambda x: x[1], reverse=True)
            data[idx]["labels"] = dict([[str(v), int(c)] for v, c in sorted_label_data])
    return data


def confusion(data, output_confusion_path: Path):
    labels = [
        "NA",
        "not a diagnosis",
        "functional",
        "muscle tension",
        "organic",
        "organic > inflammatory",
        "organic > neuro-muscular",
        "organic > structural",
        "organic > trauma",
    ]
    label_votes = dict(
        [
            [
                label,
                [
                    {
                        "NA": 0,
                        "functional": 0,
                        "muscle tension": 0,
                        "not a diagnosis": 0,
                        "organic": 0,
                        "organic > inflammatory": 0,
                        "organic > neuro-muscular": 0,
                        "organic > structural": 0,
                        "organic > trauma": 0,
                    }
                    for _ in range(7)
                ],
            ]
            for label in labels
        ]
    )
    for label in labels:
        for row in data:
            row_labels = row["labels"]
            if label in row_labels:
                idx = row_labels[label] - 1
                for key, val in row["labels"].items():
                    label_votes[label][idx][key] += val

    # print(label_votes)
    fig, ax = plt.subplots(9, 1, figsize=(10, 36), constrained_layout=True)
    for label_idx, (label, votes) in enumerate(label_votes.items()):
        vote_matrix = np.zeros((9, 7))
        for idx, vote in enumerate(votes):
            for key, val in vote.items():
                if key == "NA":
                    vote_matrix[0, idx] += val
                if key == "not a diagnosis":
                    vote_matrix[1, idx] += val
                elif key == "functional":
                    vote_matrix[2, idx] += val
                elif key == "muscle tension":
                    vote_matrix[3, idx] += val
                elif key == "organic":
                    vote_matrix[4, idx] += val
                elif key == "organic > inflammatory":
                    vote_matrix[4, idx] += val
                    vote_matrix[5, idx] += val
                elif key == "organic > neuro-muscular":
                    vote_matrix[4, idx] += val
                    vote_matrix[6, idx] += val
                elif key == "organic > structural":
                    vote_matrix[4, idx] += val
                    vote_matrix[7, idx] += val
                elif key == "organic > trauma":
                    vote_matrix[4, idx] += val
                    vote_matrix[8, idx] += val
        normalized_vote_matrix = vote_matrix / np.maximum(
            1, vote_matrix.sum(axis=0, keepdims=True)
        )
        ax[label_idx].set_title(label)
        ax[label_idx].imshow(
            normalized_vote_matrix, cmap="magma", interpolation=None, aspect="auto"
        )
        ax[label_idx].set_yticklabels(labels)
        ax[label_idx].set_yticks(range(len(labels)))
        ax[label_idx].set_xticklabels(range(8))
        ax[label_idx].set_ylabel("Label")
        ax[label_idx].set_xlabel("Vote count")
    fig.savefig(str(output_confusion_path))


def analysis(source_path: Path, output_confusion_path: Path):
    output_confusion_path.parent.mkdir(parents=True, exist_ok=True)
    data = load_labels(source_path)
    confusion(data, output_confusion_path)
