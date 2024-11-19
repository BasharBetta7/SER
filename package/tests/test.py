from caser.speech_emotions import CaserEmotionModel


er = CaserEmotionModel(model_name='caser', device='cuda')

# predict emotion from audio file:
print(er.predict_emotion(audio_path='../samples/angry​.mp3'))
