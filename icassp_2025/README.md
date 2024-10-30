# ICASSP 2025

This folder contains the work for ICASSP 2025.

## Prerequisites

1. Database
   The data for SVD has stopped auto-downloading because the online DB has gone offline. Ensure the data is available at `/home/data` and looks as following:
   ```
   root@3633b538dfed:/home# ls
   data  storage  workspace
   root@3633b538dfed:/home# ls data/
   meei  svd  voiced
   root@3633b538dfed:/home# ls data/svd
   data.json  healthy  pathological
   ```
2. Pipenv
   Install pipenv and open a pipenv shell
   ```
   root@3633b538dfed:/home/workspace/icassp_2025# pipenv install
   Loading .env environment variables...
   Installing dependencies from Pipfile.lock (566b26)...
   To activate this project's virtualenv, run pipenv shell.
   Alternatively, run a command inside the virtualenv with pipenv run.
   root@3633b538dfed:/home/workspace/icassp_2025# pipenv shell
   Loading .env environment variables...
   Loading .env environment variables...
   Launching subshell in virtual environment...
   root@3633b538dfed:/home/workspace/icassp_2025#  . /root/.local/share/virtualenvs/icassp_2025-ICtE4ZKh/bin/activate
   (icassp_2025) root@3633b538dfed:/home/workspace/icassp_2025#
   ```

## Running the existing code

1. Help for the module is available as usual

   ```
   (icassp_2025) root@3633b538dfed:/home/workspace/icassp_2025# python -m src --help
   /root/.local/share/virtualenvs/icassp_2025-ICtE4ZKh/lib/python3.10/site-packages/s3prl/upstream/byol_s/byol_a/common.py:20: UserWarning: torchaudio._backend.set_audio_backend has been deprecated. With dispatcher enabled, this function is no-op. You can remove the function call.
   torchaudio.set_audio_backend("sox_io")
   ESPnet is not installed, cannot use espnet_hubert upstream
   usage: ICASSP 2025 work [-h] {analyse,experiment,experiment_short,prepare_tasks,test,test_cross} ...

   positional arguments:
   {analyse,experiment,experiment_short,prepare_tasks,test,test_cross}

   options:
   -h, --help            show this help message and exit
   ```

2. Run an existing experiment

```
    python -m src experiment svd_speech_0_unispeechSAT
```

Checkpoints and tensorboard will be stored in `.cache` folder and can be accessed from there.

You can run tensorboard as below

```
(icassp_2025) root@3633b538dfed:/home/workspace/icassp_2025/.cache/tboard/svd_speech_0_unispeechSAT/2024_Oct_30-02_35_05-+0000# ls
events.out.tfevents.1730255705.3633b538dfed.5260.0  loss_eval  loss_train
(icassp_2025) root@3633b538dfed:/home/workspace/icassp_2025/.cache/tboard/svd_speech_0_unispeechSAT/2024_Oct_30-02_35_05-+0000# tensorboard --logdir .
TensorFlow installation not found - running with reduced feature set.

NOTE: Using experimental fast data loading logic. To disable, pass
    "--load_fast=false" and report issues on GitHub. More details:
    https://github.com/tensorflow/tensorboard/issues/4784

I1030 02:39:07.631794 139773821818432 plugin.py:429] Monitor runs begin
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.13.0 at http://localhost:6006/ (Press CTRL+C to quit)
```

## Adding / Editing stuff

You can edit `src/experiments/trainer.py` to edit the training loop, the features in `src/model/feature.py` and model in `src/model/output.py`
