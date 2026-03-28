# Introduction

Speech Emotion Recognition (SER) focuses on identifying the emotional state of a speaker from audio signals. This problem is important for applications such as human-computer interaction, voice assistants, call center analytics, and affective computing systems.

This repository implements an end-to-end pipeline for speech emotion recognition using deep learning. It covers the full workflow from data preparation and feature extraction to model training, evaluation, and inference.

The system leverages modern architectures based on PyTorch and Transformer-style models to learn robust representations from speech. It is designed to be modular and extensible, allowing experimentation with different datasets, model architectures, and training strategies.

Key capabilities of this repository include:

- Training SER models on standard emotional speech datasets
- Evaluating performance using common metrics
- Running inference on new audio samples
- Easy integration into larger pipelines or downstream applications

This project is intended both for research purposes and as a practical baseline for building real-world speech emotion recognition systems.

You can try out the model by following the instructions below, or the insructions of the README inside package/ folder.

# prepare the environment 
1. it is recommended to create a new conda environment with python 3.8
2. install required packages with 
 > pip install -r requirements.txt
# download dataset
you need to install audio files from IEMOCAP website [https://sail.usc.edu/iemocap/] and extract .wav files of all sentences of all sessions  into ./data/iemocap folder


# train the model from scratch
to train the model with the same hyperparameters as the paper, and reproduce results, simply run:
> sh train.sh 

results of cross validation will be saved as .pt files in "./checkpoints/CA_SER_fold_{fold name}.pt".

you can edit hyperparameters of the model, view  by running:
 > python ./src/run.py --help

# evaluate the model:
if you have a pre-trained model checkpoint, you can load its parameters and test it on the test set:
> python src/eval_split.py --checkpoint {CHECKPOINT_PATH}

you can download pre-trained checkpoint of the model "caser_v1.pt" from : [https://drive.google.com/file/d/1__JtzlJRF4tyH4-bjQk7a6Nw7AC-Mgau/view?usp=sharing]. After downloading, copy the path of checkpoint and paste it in --checkpoint argument above.

# Inference mode:
You can directly apply a pre-trained model on a speech audio and extract emotion label. simply type:
> python src/infer.py --checkpoint {CHECKPOINT_PATH} --audio_path {PATH_TO_AUDIO} --device {cuda | cpu}

which will output the assigned emotion to the audio, and the probability distribution over all emotion classes 
