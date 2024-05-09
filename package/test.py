from caser.speech_emotions import CaserEmotionModel


er = CaserEmotionModel(model_name='caser', device='cpu')
print(er.predict_microphone(5))
