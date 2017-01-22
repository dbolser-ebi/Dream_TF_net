# Convolutional Neural Networks for the Encode-Dream Transcription Factor Binding Challenge

## Overview
This repository contains the code for running single and multi task convolutional neural networks for the Encode-Dream Transcription Factor Binding Challenge on Synapse. This includes programmatic downloading of the data, data preprocessing, training and generating predictions.

Minimum requirements:
Python 2.7
128 GB RAM

Necessary packages can be installed using virtualenv. The requirements.txt is part of the repository.

```
$ virtualenv <env_name>
$ source  <env_name>/bin/activate
$ (<env_name>)$ pip install -r requirements.txt
```
## Data download and cleaning
In order to download the data navigate to the source folder and run:
```
$ python datadownload.py -u username -p password -l ../data/
$ python dataclean.py --path ../data/
```
Replace username and password with your synapse account username and password respectively.

## Data preprocessing
Preprocess the primary sequence and save the genome in a binary .npy file.
```
$ python datagen.py --gen_sequence --segment segment
```
Preprocess the chipseq peaks.
```
$ python datagen.py --gen_y --segment train
```
Preprocess DNASe fold change

```
$ python datagen.py --gen_dnase --segment segment --num_jobs 8
```

Replace segment with train, ladder or test to preprocess the data for the train, ladder or final segment.

## Training and generating predictions

Run and train a single task model by running the following command.
```
$ python crossval.py --tfs transcription_factor -v -l -t -m KC -c 7 -ne 5 -sbs 1000 -dbs 1000 -ntc 14 -rc --verbose
```

Replace transcription_factor with the transcription factor you want to train.
The option -v runs the internal validation benchmark, the option -l generates the leaderboard predictions (if applicable), the option -t generates the final predictions (if applicable).
The -ne option controls the number of epochs to run.
The size of the sequence window and dnase window can be controlled with the -sbs and -dbs options respectively.

Run and train a multitask (a model which trains all transcription factors simultaneously) model by running the following command.

```
$ python multicrossval.py -v -l -t -m KC -c 7 -ne 5 -sbs 1000 -dbs 1000 -ntc 14 -rc --verbose
```

Note that only the single task was used for the submission.