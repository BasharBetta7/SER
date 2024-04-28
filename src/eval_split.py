from imports import *
from run import evaluate_metrics, create_dataloaders, Config, create_processor, extract_mfcc
from models import SER2
from prepare_dataset import IEMOCAP_Dataset
@torch.no_grad()
def test_split(model, test_dataloader, device='cuda'):

    lossi = []
    ytrue = []
    ypred = []
    model.to(device)
    
    for batch_dict in tqdm(test_dataloader, total=len(test_dataloader)):
        xtr_1 = batch_dict['batch_audio']
        xtr_2 = batch_dict['batch_mfcc']
        ytr = batch_dict['batch_labels']
       
        
        # forward pass:
        logits, loss =  model(xtr_1.to(device), xtr_2.to(device),  ytr.to(device))
        ypred.extend(logits.detach().cpu().argmax(1))
        ytrue.extend(ytr.detach().cpu())
        lossi.append(loss.detach().cpu())
        
    wa, ua = evaluate_metrics(ypred, ytrue)
    
    return {'WA':wa, 'UA':ua}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='SER_CA_attention', description='evaluate Speech Emotion Recognition Model on IEMOCAP dataset')
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="checkpoint .pt file path"
    )
    parser.add_argument(
        "--config", type=str, default='./configs/config.yaml', help="checkpoint .pt file path"
    )
    args = parser.parse_args()
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = SER2(40, 512,512,4,256)
    model.load_state_dict(torch.load(args.checkpoint)['model_state_dict'])
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
        
  
    model_config = Config(config["COSER"])
    dataset_config = Config(config["DATASET"])
    
    PATH_TO_WAV = dataset_config.wav_path
    PATH_TO_CSV = dataset_config.wav_csv

    ### load the dataset
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    data = pd.read_csv(PATH_TO_CSV)
    test_size = int(0.1 * len(data))
    idx = torch.multinomial(torch.ones((len(data))), num_samples=test_size, replacement=False) # sample from uniform distribution 
    test_data_csv = data.iloc[idx].copy()
    test = test_data_csv.reset_index()

    test= IEMOCAP_Dataset(dataset_config, test)
    audio_processor = create_processor("facebook/wav2vec2-base")

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
            np.array([extract_mfcc(audio, dataset_config.sr) for audio in batch_audio])
        )

        batch_audio = audio_processor(
            batch_audio, sampling_rate=16000, return_tensors="pt"
        ).input_values[0]

        return {
            "batch_audio": batch_audio,
            "batch_mfcc": mfcc,
            "batch_labels": torch.tensor(target_labels, dtype=torch.long),
        }
    test_dataloader = DataLoader(test, batch_size=1,shuffle=False,collate_fn=collate)
    print(test_split(model, test_dataloader=test_dataloader))