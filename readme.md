# prepare the environment 
1. it is recommended to create a new conda environment with python 3.8
2. install required packages with 
 > pip install -r requirements.txt
# download dataset
you need to install audio files from IEMOCAP website [https://sail.usc.edu/iemocap/] and extract .wav files of all sentences of all sessions  into ./data/iemocap folder
# reproduce the results

# train the model from scratch
to train the model with the same hyperparameters as the paper, simply run:
> sh train.sh 

results of cross validation will be saved as .pt files in ./checkpoints/CA_SER_fold_{fold name}.

you can view hyperparameters by running:
 > python ./src/run.py --help

# evaluate the model:
if you have a pre-trained model checkpoint, you can load its parameters and test it on the test set:
> python src/eval_split.py --checkpoint {CHECKPOINT_PATH}

# Inference mode:
You can directly apply a pre-trained model on a speech audio and extract emotion label. simply type:
> python src/infer.py --checkpoint {CHECKPOINT_PATH} --audio_path {PATH_TO_AUDIO} --device {cuda | cpu}

which will output the assigned emotion to the audido, and the probability distribution over all emotion classes 