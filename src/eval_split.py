from imports import *
from run import evaluate_metrics, create_dataloaders, Config
from models import SER2

@torch.no_grad()
def test_split(model, split, train:DataLoader, val:DataLoader, test:DataLoader, device='cuda'):
    loader = {
        'train': train,
         'val' : val,
         'test' : test
     }[split]


    lossi = []
    ytrue = []
    ypred = []
    model = model
    model.to(device)
    
    for batch_dict in tqdm(loader, total=len(loader)):
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
    tr,val,test = create_dataloaders(model_config, dataset_config, args, 'Ses01')
    print(test_split(model, args.split, tr, val, test))