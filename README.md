# Trustworthy Machine Learning, TAU, Spring 2024 - SOPHON-based adaptation for the speech domain
## Overview
This project implements the method introduced by the technical report submitted for the Trustworthy Machine Learning course project done in Tel Aviv University, Spring 2024
of an adaptation based on the SOPHON approach for non-fine-tunability of pretrained models in the speech domain.
The method consists of two modules, a suppression module and a reinforcement module, the suppression module degrades the performance on the restricted domains,
and the reinforcement module aims to improve the performance of the model so that the suppression module won't degrade the general performance of the model too much.

## Steps for Reproducing Results
### Environment Requirements
* Download the [fairseq]([https://github.com/s3prl/s3prl/tree/main/s3prl/downstream](https://github.com/facebookresearch/fairseq)) framework
* Download the [s3prl]([https://github.com/s3prl/s3prl](https://github.com/s3prl/s3prl/tree/main/s3prl/downstream)) framework
* Replace `runner.py` file in the `s3prl` framework with ours.
