from caser.imports import *


 ### labels =  data['Label'].unique()
label_to_index = {
        'hap': 0,
        'ang': 1,
        'neu': 2,
        'sad': 3,
        'exc': 0
    }

index_to_label = {v:k for k,v in label_to_index.items()}
index_to_label[0] = 'hap'
index_to_label

# we merge 'hap' and 'exc' together in a single class 
class IEMOCAP_Dataset():
    def __init__(self,config, data_csv):
        self.audio_features = 'raw'
        self.base_dir = config.wav_path
        self.data_csv = data_csv

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
        if wav.shape[0] > 208000: #limit audio length to 13 seconds 
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

# create train/val/test datasets  ~ 10/10/80 :
# test set is randomly selected from dataset :
def prepare_dataset_csv(config, valid_session='Ses01', test_rate=0.1):

    PATH_TO_WAV = config.wav_path
    PATH_TO_CSV = config.wav_csv

    ### load chunk of the dataset

    print('checking integrity of .csv file...')
    data = pd.read_csv(PATH_TO_CSV)
    for _,_, files in os.walk(PATH_TO_WAV):
        print(f"all .wav files inside \"{PATH_TO_WAV}\" exist in .csv file \"{PATH_TO_CSV}\":",  sum([file.split(sep='.')[0] in data['FileName'].values for file in files]) == data['FileName'].count())
        print(f"all filenames inside .csv file {PATH_TO_CSV} exists  in {PATH_TO_WAV}:", sum([file+'.wav' in files for file in data['FileName'].values]) == len(files))
  
    
    # '''test set is randomly selected, and validation is taken from valid_session'''


    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    test_size = int(test_rate * len(data))
    if test_size > 0:
        idx = torch.multinomial(torch.ones((len(data))), num_samples=test_size, replacement=False) # sample from uniform distribution 
        test_data_csv = data.iloc[idx].copy()
        valid_data_csv = data[data['FileName'].str.match(valid_session) & ~data.index.isin(test_data_csv.index)]
        train_data_csv = data[~data.index.isin(valid_data_csv.index) & ~data.index.isin(test_data_csv.index)]
        
        test_data_csv = test_data_csv.reset_index()
        valid_data_csv = valid_data_csv.reset_index()
        train_data_csv = train_data_csv.reset_index()
        return train_data_csv, valid_data_csv, test_data_csv
    else:
        valid_data_csv = data[data['FileName'].str.match(valid_session)]
        train_data_csv = data[~data.index.isin(valid_data_csv.index)]
        valid_data_csv = valid_data_csv.reset_index()
        train_data_csv = train_data_csv.reset_index() 
        return  train_data_csv, valid_data_csv