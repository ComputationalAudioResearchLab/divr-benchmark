# DiVR Benchmark Tasks (v1)

First revision of benchmarking tasks for disordered voice recognition. These sets of tasks are aimed to answer a few key questions of the domain, as well as provide a standardize measurement metric for comparing future work.

### Overall specifications

1. Only publicly available datasets (SVD, Voiced and Torgo) are considered.
2. It uses the diagnosis_map_v1.
3. All sessions classified as unclassified and unclassified_pathology are ignored. Except in case of Torgo where we only have dysarthria for which we don't have a hierarchical classification.
4. Only audio data is considered.
5. Only diagnosis levels 0, 1 and 2 are considered.
6. The test set is restricted to 16kHz sample rate, z-score normalized audio.
7. If multiple diagnosis are available for any given data point, then as long as the predicted diagnosis matches any one of the diagnosis then it's considered a success.

## Task Streams

### 0. Sanity checking

This stream is to do level 0 classification i.e. binary healthy vs pathological classificaiton. We have included this in this bencmark only because the test is present extensively in existing literature and can serve as a useful sanity check for experiments. The train and validation set for this stream consists of all data from SVD, Voiced and Torgo and there are following four test sets:

1. SVD - level 0 - /a/, /i/, /u/
2. SVD - level 0 - connected speech
3. Voiced - level 0 - /a/
4. Torgo - level 0 - connected speech

### 1. Minimum and sufficient vocal task

This stream aims to identify which vocal tasks we need patients to perform in order to get an acceptable level of recognition accuracy. We validate this at all levels of classification. Since out of the public datasets only SVD has multiple vocal tasks available this stream is limited to only using SVD. There are following 10 tasks that contain train, validation and test set respectively:

1. level 1 - /a/
2. level 1 - /i/
3. level 1 - /u/
4. level 1 - /a/, /i/, /u/
5. level 1 - connected speech
6. level 2 - /a/
7. level 2 - /i/
8. level 2 - /u/
9. level 2 - /a/, /i/, /u/
10. level 2 - connected speech

### 2. Cross domain transfer

This task stream is to evaluate how well a fully trained model can transfer from one domain to another. We use the entire SVD dataset as that has the maximum amount of data for training in this stream and test on Voiced and Torgo. The following tests sets are available:

1. level 0 - voiced
2. level 1 - voiced
3. level 2 - voiced
4. level 1 - torgo

### 3. Low resource training

As Voiced and Torgo have significantly less data than SVD we believe this provides an interesting avenue to test low resource ML algorithms. This task is restricted to only training on data from Voiced and Torgo, but is tested on all 3 datasets with the test sets listed below. The training set provides level 2 classifications however they can easily be converted to level 1 classifications using the benchmark library if needed for experiments.

1. level 1 - voiced
2. level 1 - torgo
3. level 1 - svd
4. level 2 - voiced
5. level 2 - svd
