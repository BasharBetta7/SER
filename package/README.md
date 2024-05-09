# Install the package:
clone the repository into your machine, then navigate to package/ and run:
> python setup.py install

or:
> pip install caser

which will add the package to your python environment.

# Usage instructions:
first you need to include 'caser' model:
> from caser.speech_emotions import CaserEmotionModel

Then, we define a new model:
> emotion_recognizer = CaserEmotionModel(model_name='caser', device='cpu')

## classify audio from path:
if you have an audio file to classify, you need to call the function predict_emotion() as follows:
> print(emotion_recognizer.predict_emotion(audio_path))

## classify audio recorded from microphone:
> print(emotion_recognizer.predict_microphone(duration))
