# Tasks

This set of tasks is designed to test various hypotheses. While initially intended as a benchmark for future research, the results indicate that robust multi-class voice disorder classification remains a distant goal. As such, establishing a comprehensive benchmark at this stage is impractical. However, these tasks provide a foundation for future work to build upon, ultimately contributing to the development of a meaningful benchmark for multi-class voice disorder classification.

## Same-Database Tests

These tasks are based solely on the SVD dataset, as it offers the greatest variety of audio tasks and pathologies found in various diagnostic classification systems.

All generated data is stored with its narrow classification labels—i.e., the labels provided by the database itself—rather than any further subclassifications. The classification systems listed below determine which diagnoses are included in each task, as different systems vary in data availability. More details on dataset availability can be found in [Counts Per Speaker](../.docs/counts_per_speaker.md).

### USVAC 2025

Tasks generated using the USVAC 2025 classification system, which comprehensively covers the dataset’s various labels and was designed by clinicians:

- **a_n**: Neutral /a/ vowel
- **i_n**: Neutral /i/ vowel
- **u_n**: Neutral /u/ vowel
- **phrase**: Connected speech
- **all**: Combination of neutral /a/, /i/, and /u/ vowels, plus connected speech

### Other Classification Systems

Tasks are also generated using the following classification systems. Since these systems cover fewer diagnoses, the resulting datasets are smaller than those generated using USVAC 2025:

- **Compton 2022**
- **da Silva Moura 2024**
- **Sztaho 2018**
- **Zaim 2023**

In addition to the standard **a_n, i_n, u_n,** and **phrase** datasets, we generate a **phrase-with-unclassified** dataset. This dataset includes all data from the USVAC 2025-based phrase dataset, but any diagnoses not present in the given classification system are marked as "unclassified" and retained in the dataset.

## Cross-Database Tests

These datasets do not have separate training and validation sets. Instead, all data from each dataset is merged into a single test set for evaluation purposes.

- **avfad**
- **meei**
- **torgo**
- **uaspeech**
- **uncommon_voice**
- **voiced**

Notably, **SVD is excluded** from this list since it is used for training in the same-database tests. Models trained on SVD will be evaluated against these external datasets.
