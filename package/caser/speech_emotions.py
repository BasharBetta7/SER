from caser.imports import *
from caser.run import *
from caser.models import SER2_transformer_block
from caser.prepare_dataset import *
import gdown
import sounddevice as sd

def find_path(model_name):
    model_file=model_name+'.pt'

    cache_dir = os.path.join(os.path.expanduser('~'), '.caser')
    os.makedirs(cache_dir, exist_ok=True)
    fpath=os.path.join(cache_dir,model_file)
    if not os.path.isfile(fpath):
        print('Downloading pre-trained model..')
        id = '1__JtzlJRF4tyH4-bjQk7a6Nw7AC-Mgau'
        gdown.download(id=id, output=fpath)
    return fpath     
   


class CaserEmotionModel:
    def __init__(self, model_name='caser', device='cpu'):
        self.device = device
        self.model= SER2_transformer_block(40, 512,512,8,256, 4)
        model_path = find_path(model_name)
        print(model_path)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        self.audio_processor = audio_processor
        
        
    def predict_microphone(self, duration):
        # Record audio
        sample_rate = 16000
        print('Start talking...')
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()  # Wait until recording is finished
        print('end')

        # Convert audio to a tensor
        audio_tensor = np.squeeze(audio)
        return self.predict_emotion_multi([audio_tensor])


    def extract_features(self, wav):
        '''extracts wav data from path '''
       
        wav = torch.tensor(wav)
        if wav.shape[0] > 208000: #limit audio length to 13 seconds 
                wav = wav[:208000]

        if wav.shape[0] < 3 * 16000:
            extended = wav.clone()       
            while extended.shape[0] < 3 * 16000: # extend small audio files       
                extended = torch.cat((extended,wav), dim=0)
            wav = extended.clone()
        
        wrapper = {'audio_input': wav, 'label': 0}
        return wrapper
    
    
    def collate(self, batch_sample):
            # batch_sample is dict type, we want to output [X, y]
            batch_audio = [X["audio_input"] for X in batch_sample]
            target_labels = [X["label"] for X in batch_sample]

            batch_audio = [{"input_values": audio} for audio in batch_audio]
            batch_audio = audio_processor.pad(
                batch_audio,
                padding=True,
                return_tensors="pt",
            ).input_values

            mfcc = torch.tensor(
                np.array([extract_mfcc(audio, 16000) for audio in batch_audio])
            )

            batch_audio = audio_processor(
                batch_audio, sampling_rate=16000, return_tensors="pt"
            ).input_values[0]

            return {
                "batch_audio": batch_audio,
                "batch_mfcc": mfcc,
                "batch_labels": torch.tensor(target_labels, dtype=torch.long),
            }
    
    
        
    # predict emotion for an audio file from path

    def predict_emotion(self, audio_path, return_logits=True):
        wav, sr = librosa.load(audio_path, sr=16000)
        features = self.extract_features(wav)     
        
        # create dataloaders
        dl = DataLoader(
            dataset=[features],
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=self.collate,
        )
        logits = None
        for i, batch_dict in enumerate(dl):
                xtr_1 = batch_dict["batch_audio"]
                xtr_2 = batch_dict["batch_mfcc"]
                ytr = batch_dict["batch_labels"]

                # forward pass:
                logits, loss = self.model(xtr_1.to(self.device), xtr_2.to(self.device), ytr.to(self.device))
       
        index_to_name = {'hap': 'Happy', 'neu': 'Neutral', 'sad': 'Sad', 'ang':'Angry'} 
       
        if return_logits:
            
            return {'Prediction':index_to_name[index_to_label[logits.argmax().item()]], 'logits': {index_to_label[k]:f'{logits[0,k].item():.4f}' for k in index_to_label.keys()}}
        else:
            return {'Prediction':index_to_name[index_to_label[logits.argmax().item()]]}
        
    # predict emotions of list of audio files 
    def predict_emotion_multi(self, audio_wav:list, return_logits=True):
        '''audio_wav is list of audio values'''
        
        features = [self.extract_features(w) for w in audio_wav]
        
        # create dataloaders
        dl = DataLoader(
            dataset=features,
            batch_size=4,
            shuffle=False,
            num_workers=1,
            collate_fn=self.collate,
        )
        logits_list = []
        ypred = []
        
        for i, batch_dict in enumerate(dl):
                xtr_1 = batch_dict["batch_audio"]
                xtr_2 = batch_dict["batch_mfcc"]
                ytr = batch_dict["batch_labels"]

                # forward pass:
                logits, loss = self.model(xtr_1.to(self.device), xtr_2.to(self.device), ytr.to(self.device))
      
                logits_list.extend(logits)
                ypred.extend(logits.detach().cpu().argmax(1))
                
        index_to_name = {'hap': 'Happy', 'neu': 'Neutral', 'sad': 'Sad', 'ang':'Angry'} 

        if return_logits:
            
            return {'Prediction':[index_to_name[index_to_label[logits.argmax().item()]] for logits in logits_list], 'logits': [logits for logits in logits_list]}
        else:
            return {'Predictions':[index_to_name[index_to_label[logits.argmax().item()]] for logits in logits_list] }


