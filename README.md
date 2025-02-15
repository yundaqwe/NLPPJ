# Trustworthy Machine Learning, TAU, Spring 2024 - SOPHON-based adaptation for the speech domain
## Overview
This project implements the method introduced by the technical report submitted for the Trustworthy Machine Learning course project done in Tel Aviv University, Spring 2024
of an adaptation based on the SOPHON approach for non-fine-tunability of pretrained models in the speech domain.
Informally, the method consists of two modules, a suppression module and a reinforcement module, the suppression module degrades the performance on the restricted domains,
and the reinforcement module aims to improve the performance of the model so that the suppression module won't degrade the general performance of the model too much.

## Steps for Reproducing Results
* Install the [fairseq]( https://github.com/facebookresearch/fairseq ) framework (`pip install fairseq`)
* Download the [s3prl]( https://github.com/s3prl/s3prl/tree/main/s3prl/downstream ) framework (via cloning)
* Replace `runner.py` file in the `s3prl` framework with ours.
* Move `sophon` folder to the `s3prl/s3prl/downstream/` folder in the s3prl clone.
* See the [General Usage](https://github.com/s3prl/s3prl/tree/main/s3prl/downstream#general-usage) section in the `s3prl` documentation to understand how to run the experiments

## Key Points
* See [config.yaml](./sophon/config.yaml) for the configuration in which we execute the runner (including the injections of parameters used in the experiments, number of suppression/reinforcement steps, etc.).
* Note that some of the files in the `sophon` folder are copied and rewritten to fit our needs, such as the [hubert_dataset.py](./sophon/hubert_dataset.py).
* For the implementationof of the suppression and refinforcement parts of our algorithm, see the `domainSuppression` and `pretrainedModelReinforcement` methods in the `runner.py`file .
