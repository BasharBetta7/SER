from imports import *
from models import SER2_transformer_block_alpha
import random
from run import train_run, prepare_dataset_csv,  collate_2, Config, create_dataloaders, IEMOCAP_Dataset
from prepare_dataset import label_to_index, index_to_label


param_distribution = {
    'd_mfcc' : [256, 512,1024],
'd_wav' : [256, 512, 1024],
'n_mfcc' : [26, 40, 60],
'coheads' : [2,4,8],
'embed_dim' : [128, 256, 512],
'num_encoders' : [1,2,3,4],
'n_heads_encoder' : [1,2,4],
    'alpha' : [0, 0.1, 0.4, 0.7, 1],
    'batch_size' : [2, 4, 6],
}



hyperparams = {k:random.choice(param_distribution[k]) for k in param_distribution.keys()}
hyperparams
def random_search(model, num_iterations, param_distribution):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    best_acc = 0.0
    best_params = None
    for i in range(num_iterations):
        hyperparams = {k:random.choice(param_distribution[k]) for k in param_distribution.keys()}
        tr_dl = DataLoader(dataset = train_dataset, batch_size = hyperparams['batch_size'], collate_fn= collate_2, num_workers = 1, shuffle = True)
        val_dl = DataLoader(dataset = valid_dataset, batch_size = hyperparams['batch_size'], collate_fn= collate_2, num_workers = 1, shuffle = False)
        device= 'cuda'
        print(f'start grid search #{i+1} with hyperparams:\n{hyperparams}')
        model.__init__(40, hyperparams['d_mfcc'], hyperparams['d_mfcc'], hyperparams['coheads'], hyperparams['embed_dim'], hyperparams['num_encoders'], hyperparams['n_heads_encoder'], n_labels=4, alpha= hyperparams['alpha'])
        wa = train_run(model,model_config, 5, tr_dl, val_dl, hyperparams['batch_size'], 4)
        print(f'WA for it {i+1} : {wa}')
        if wa > best_acc :
            best_acc = wa
            best_params = hyperparams
    return {'best_wa': best_acc, 'best_params': best_params}
    
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='SER_Co_attention', description='train Speech Emotion Recognition Model on IEMOCAP dataset')
    parser.add_argument(
        "--config", type=str, default='./configs/config.yaml', help="configuration .yaml file path"
    )
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
        
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model_config = Config(config["COSER"])
    dataset_config = Config(config["DATASET"])
  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SER2_transformer_block_alpha(40, 512,512,4,256,2,4)

    
    train, valid, test = prepare_dataset_csv(dataset_config)
    train_dataset = IEMOCAP_Dataset(dataset_config, train)
    valid_dataset = IEMOCAP_Dataset(dataset_config, valid)
    results = random_search(model, 10, param_distribution)
    
