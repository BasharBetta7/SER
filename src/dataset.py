from imports import *

class IEMOCAP_Dataset():
    def __init__(self, data_csv=None):
        self.audio_features = 'raw'
        self.base_dir = PATH_TO_WAV
        self.data_csv = data_csv if data_csv is not None else data

    def __len__(self):
        return len(self.data_csv)

    def extract_mfcc(self, wav, sr):
        '''wav : (1,T)'''
        x= np.array(wav)
        x = librosa.effects.preemphasis(x, zi = [0.0])
        hop_length = int(0.01 * sr) #10ms sequences
       
        mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40, hop_length=hop_length, htk=True).T
            
        return mfcc
        

    def __getitem__(self, index):
        
        wav, sr = torchaudio.load(os.path.join(self.base_dir, self.data_csv['FileName'][index]+'.wav'))
        wav = wav.squeeze(0)
        if wav.shape[0] > 208000:
            wav = wav[:208000]

        if wav.shape[0] < 3 * sr:
            extended = wav.clone()       
            while extended.shape[0] < 3 * sr: # extend small audio files       
                extended = torch.cat((extended,wav), dim=0)
            wav = extended.clone()
            
        
        mfccs = self.extract_mfcc(wav, sr)
        # wav, sr = librosa.core.load(audio_path + ".wav", sr=None)
        label = label_to_index[self.data_csv.iloc[index].Label]
        features = None
        if self.audio_features == 'raw':
            features = wav
        
        return {'audio_input':wav, 'audio_length': wav.shape[0], 'label' :label, 'mfcc' : mfccs}