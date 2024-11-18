from caser.speech_emotions import CaserEmotionModel


er = CaserEmotionModel(model_name='caser', device='cuda')
print(er.predict_emotion(''))
