from imports import *
from run import audio_processor, evaluate_metrics, create_dataloaders,  Config, create_processor, extract_mfcc
from models import SER2_transformer_block
from prepare_dataset import IEMOCAP_Dataset,index_to_label, label_to_index


@torch.no_grad()
def infer_model(model:SER2_transformer_block, audio_path:str, device):
    '''
    model: pre-trained model from checkpoint
    '''
    # load the model
    # perform a forward pass
   # assert audio_path[-3:-1] == 'wav', print('audio path must be in wav format')
    
    model.to(device)
    wav, sr = librosa.load(audio_path, sr=16000)
    wav = torch.tensor(wav)
    if wav.shape[0] > 208000: #limit audio length to 13 seconds 
            wav = wav[:208000]

    if wav.shape[0] < 3 * sr:
        extended = wav.clone()       
        while extended.shape[0] < 3 * sr: # extend small audio files       
            extended = torch.cat((extended,wav), dim=0)
        wav = extended.clone()
    
    assert(sr == 16000), print('audio file must be sampled at 16Khz')
    wrapper = {'audio_input': wav, 'label': 0}
    def collate(batch_sample):
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
            np.array([extract_mfcc(audio, sr) for audio in batch_audio])
        )

        batch_audio = audio_processor(
            batch_audio, sampling_rate=16000, return_tensors="pt"
        ).input_values[0]

        return {
            "batch_audio": batch_audio,
            "batch_mfcc": mfcc,
            "batch_labels": torch.tensor(target_labels, dtype=torch.long),
        }


    # create dataloaders
    dl = DataLoader(
        dataset=[wrapper],
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=collate,
    )
    logits = None
    for i, batch_dict in enumerate(dl):
            xtr_1 = batch_dict["batch_audio"]
            xtr_2 = batch_dict["batch_mfcc"]
            ytr = batch_dict["batch_labels"]

            # forward pass:
            logits, loss = model(xtr_1.to(device), xtr_2.to(device), ytr.to(device))
            
    index_to_name = {'hap': 'Happy', 'neu': 'Neutral', 'sad': 'Sad', 'ang':'Angry'} 

    return {'Prediction':index_to_name[index_to_label[logits.argmax().item()]], 'logits': {index_to_label[k]:f'{logits[0,k].item():.4f}' for k in index_to_label.keys()}}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='CA_SER Model', description='Infer with an audio file')
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="checkpoint .pt file path"
    )
    parser.add_argument(
        "--audio_path", type=str, required=True, help="path of audio .wav file to classify"
    )
    parser.add_argument("--device", type=str, default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = SER2_transformer_block(40, 512,512,4,256);
    model.load_state_dict(torch.load(args.checkpoint)['model_state_dict']);
    results = infer_model(model, args.audio_path, args.device)
    
    print(results)