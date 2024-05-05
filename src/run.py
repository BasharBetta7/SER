# prepare dataset
# define the model
# define hyperparameters
# create dataloaders
# define the metrics
# train the model
# extract results : accuracy plots, confusion matrix
# save results in external file

from imports import *
from models import *
from prepare_dataset import prepare_dataset_csv, IEMOCAP_Dataset



class Config:
    def __init__(self, config_dict):
        for (
            k,
            v,
        ) in config_dict.items():
            setattr(self, k, v)


def extract_mfcc(wav, sr):
    """wav : (1,T)"""
    x = np.array(wav)
    x = librosa.effects.preemphasis(x, zi=[0.0])
    hop_length = int(0.01 * sr)  # 10ms sequences

    mfcc = librosa.feature.mfcc(
        y=x, sr=sr, n_mfcc=40, hop_length=hop_length, htk=True
    ).T

    return mfcc


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

    batch_audio = audio_processor(
        batch_audio, sampling_rate=16000, return_tensors="pt"
    ).input_values[0]

    return (batch_audio, torch.tensor(target_labels, dtype=torch.long))


def create_processor(wav2vec_config):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_config)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        wav2vec_config, do_lower_case=False, word_delimiter_token="|"
    )
    return Wav2Vec2Processor(feature_extractor, tokenizer)


audio_processor = create_processor("facebook/wav2vec2-base")
def collate_2(batch_sample):  # extracts mfcc"
    # batch_sample is dict type, we want to output [X, y]
    batch_audio = [X["audio_input"] for X in batch_sample]
    target_labels = [X["label"] for X in batch_sample]

    batch_audio = [{"input_values": audio} for audio in batch_audio]
    batch_audio = audio_processor.pad(
        batch_audio,
        padding=True,
        return_tensors="pt",
    ).input_values

    mfcc = torch.tensor(np.array([extract_mfcc(audio, sr) for audio in batch_audio]))

    batch_audio = audio_processor(
        batch_audio, sampling_rate=16000, return_tensors="pt"
    ).input_values[0]

    return {
        "batch_audio": batch_audio,
        "batch_mfcc": mfcc,
        "batch_labels": torch.tensor(target_labels, dtype=torch.long),
    }


def create_dataloaders(
    model_config: Config, dataset_config: Config, args, valid_session
):
    """return train, validation dataloaders"""
    train, val, test = prepare_dataset_csv(
        dataset_config, valid_session=valid_session, test_rate=0.1
    )
    train_dataset = IEMOCAP_Dataset(dataset_config, train)
    valid_dataset = IEMOCAP_Dataset(dataset_config, val)
    test_dataset = IEMOCAP_Dataset(dataset_config, test)

    # HYPERPARAMETERS:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size
    num_workers = 1
    accum_iter = int(
        args.accum_grad
    )  # accumulate grads (maybe needs search for optimal value)
    sr = int(dataset_config.sr)

    print(f"SSL MODEL: {model_config.ssl_model}")
    if model_config.ssl_model == "wavlm":
        audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-base-plus"
        )
    elif model_config.ssl_model == "wav2vec":
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

    print("CREATING DATALOADERS...")
    # create dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
    )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate,
    )
    
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate,
    )

    return train_dataloader, valid_dataloader, test_dataloader


def evaluate_metrics(ypred, ytrue):
    # ypred : (B), ytrue : (B)
    ypred = torch.tensor(ypred, dtype=torch.float32)
    ytrue = torch.tensor(ytrue, dtype=torch.float32)
    ua = (ypred == ytrue).float().mean()
    ypred_onehot = torch.eye(4)[ypred.long()]
    ytrue_onehot = torch.eye(4)[ytrue.long()]
    wa = torch.mean(
        torch.sum((ypred_onehot == ytrue_onehot) * ytrue_onehot, 0)
        / torch.sum(ytrue_onehot.int(), 0)
    )
    return wa.item(), ua.item()


class  EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0, type='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float("inf") 
        self.type = type

    def early_stop(self, loss):
        if self.type == 'max':
            loss *= -1
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss > (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_run(
    model,
    config,
    epochs,
    train_dl,
    valid_dl,
    batch_size,
    accum_iter,
    early_stopping:EarlyStopping=None,
):
    model.to(device)
    params_wav = [p for p in model.wav2vec_encoder.parameters() if p.requires_grad]
    params_wav_ids = {id(p) for p in params_wav}
    params_other = [p for p in model.parameters() if p.requires_grad and id(p) not in params_wav_ids]
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"Model : {model.__class__.__name__}")
    print(
        f"Total Parameters: {sum(p.numel() for p in model.parameters())}\n Learnable parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}\n"
    )
    print("*" * 50)
    torch.cuda.empty_cache()
    gc.collect()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    # optimizer = torch.optim.AdamW([{'params':params_wav, 'lr':config.lr}, {'params':params_other, 'lr':config.lr_other}])
    optimizer = torch.optim.AdamW(params, lr=config.lr)
    optimizer.zero_grad(set_to_none=True)
    scheduler = ReduceLROnPlateau(
        optimizer, "max", patience=5, min_lr=1e-8, verbose=True
    )
    logs_freq = 50

    b_loss_val_i = []
    blossi = []  # batch loss i : a list of batch losses
    logs = ""
    wa_val = []
    ua_val = []
    wa_tr = []
    ua_tr = []
    max_epochs = epochs if epochs > 0 else config.epochs

    wa_best = 0.0
    chkpt = {}

    print("START TRAINING:")
    for step in range(max_epochs):
        print(f"epoch {step + 1}\n")
        start_time = time.time()
        ypred = []
        ytrue = []
        model.train()
        time.sleep(2)
        for i, batch_dict in enumerate(train_dl):
            xtr_1 = batch_dict["batch_audio"]
            xtr_2 = batch_dict["batch_mfcc"]
            ytr = batch_dict["batch_labels"]

            # forward pass:
            logits, loss = model(xtr_1.to(device), xtr_2.to(device), ytr.to(device))

            blossi.append(loss.detach().cpu().item())
            loss = loss / accum_iter
            ypred.extend(logits.detach().cpu().argmax(1))
            ytrue.extend(ytr.detach().cpu())
            # backward pass
            loss.backward()
            if ((i + 1) % accum_iter == 0) or ((i + 1) == len(train_dl)):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                # running_loss=0

            if (i + 1) % logs_freq == 0:
                print(
                    f"Batch {i+1}/{len(train_dl)} | Train loss : {torch.tensor(blossi[-logs_freq:]).mean(0).detach().cpu().numpy()}"
                )

        loss_epoch_mean = (
            torch.tensor(blossi[-len(train_dl) :]).mean(0).detach().cpu().numpy()
        )

        wa_train, ua_train = evaluate_metrics(ypred, ytrue)
        wa_tr.append(wa_train)
        ua_tr.append(ua_train)
        print(
            f"epoch {step + 1} Train WA= {wa_train} |  UA={ua_train} |  loss={loss_epoch_mean}"
        )

        ypred = []
        ytrue = []
        model.eval()
        time.sleep(2)
        with torch.no_grad():
            for batch_dict in tqdm(valid_dl, total=len(valid_dl)):
                xval_1 = batch_dict["batch_audio"]
                xval_2 = batch_dict["batch_mfcc"]
                yval = batch_dict["batch_labels"]
                logits, loss = model(
                    xval_1.to(device), xval_2.to(device), yval.to(device)
                )
                ypred.extend(logits.detach().cpu().argmax(1))
                ytrue.extend(yval.detach().cpu())
                b_loss_val_i.append(loss.detach().cpu().item())

            wa, ua = evaluate_metrics(ypred, ytrue)
            wa_val.append(wa)
            ua_val.append(ua)
            val_loss = (
                torch.tensor(b_loss_val_i[-len(valid_dl) :])
                .mean(0)
                .detach()
                .cpu()
                .numpy()
            )
            elapsed_time = time.time() - start_time
            scheduler.step(wa)
            print(f"Current learning rate: {scheduler.get_last_lr()}")
            print(f"epoch {step + 1} validation WA= {wa} | UA={ua} | loss={val_loss}")
            stats = {
                "epoch": step,
                "wa": wa,
                "ua": ua,
                "val_loss": val_loss.item(),
                "train_loss": loss_epoch_mean.item(),
                "time_elapsed": elapsed_time,
            }

            logs += f"{json.dumps(stats)}\n"

            # if early_stopping(torch.tensor(b_loss_val_i).view(len(valid_dl),-1).mean(0), 3,1e-3):
            #     print('early stopping')
            #     break

            # save best model:
            if wa > wa_best:
                wa_best = wa
                chkpt = model.state_dict().copy()
            if early_stopping is not None:
                if early_stopping.early_stop(wa):
                    print("Eearly Stopping!")
                    break
    model.to("cpu")
    print("END TRAINING")
    return {
        "model_name": model.__class__.__name__,
        "train_loss_per_batch": blossi,
        "validation_loss_per_batch": b_loss_val_i,
        "validation_wa_per_epoch": wa_val,
        "validation_ua_per_epoch": ua_val,
        "train_wa_per_epoch": wa_tr,
        "train_ua_per_epoch": ua_tr,
        "logs": logs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_chkpt": chkpt,
    }


def cross_validation_5_folds(model,model_args:tuple, model_config:Config, dataset_config:Config, args):
    cv_results = []
    print("START CROSS-VALIDATION...")
    model.__init__(*model_args)
    for i in range(1, 6):

        print(f"Fold #{i}")
        print("PREPARING DATASET...")
        train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(
            model_config, dataset_config, args, valid_session=f"Ses0{i}"
        )

        print("INITALIZING THE MODEL...")
        gc.collect()
        torch.cuda.empty_cache()
        model.__init__(*model_args)

        es = None
        if args.early_stop:
            es = EarlyStopping(patience=7, min_delta=0.0, type='max')
            print("Early Stopping is activated")
        results = train_run(
            model=model,
            config=model_config,
            epochs=args.epochs,
            train_dl=train_dataloader,
            valid_dl=valid_dataloader,
            batch_size=args.batch_size,
            accum_iter=args.accum_grad,
            early_stopping=es,
        )
        cv_results.append(results)
        torch.save(
            results,
            f"{args.save_path}/{model.__class__.__name__}_{args.model_name}_fold_{i}.pt",
        )
        print(f'stats of the fold are saved in P{args.save_path}/{model.__class__.__name__}_{args.model_name}_fold_{i}.pt')
    torch.save(
        cv_results,
        f"{args.save_path}/{model.__class__.__name__}_{args.model_name}_cv_results_5.pt",
    )
    print(f"stats are saved in {args.save_path}/{model.__class__.__name__}_{args.model_name}_cv_results_5.pt")
    
    
def cross_validation_10_folds(model:nn.Module,model_args:tuple, model_config:Config, dataset_config:Config, args):
    cv_results = []
    print("START CROSS-VALIDATION...")
    
    model.__init__(*model_args)
    for i in '1F,2M,2F,3M,3F,4M,4F,5M,5F'.split(sep=','):
        print(f"Fold #{i}")
        print("PREPARING DATASET...")
        train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(
            model_config, dataset_config, args, valid_session=f"Ses0{i}"
        )

        print("INITALIZING THE MODEL...")
        gc.collect()
        torch.cuda.empty_cache()
        model.__init__(*model_args)

        es = None
        if args.early_stop:
            es = EarlyStopping(patience=7, min_delta=1e-3, type='max')
            print("Early Stopping is activated")
        results = train_run(
            model=model,
            config=model_config,
            epochs=args.epochs,
            train_dl=train_dataloader,
            valid_dl=valid_dataloader,
            batch_size=args.batch_size,
            accum_iter=args.accum_grad,
            early_stopping=es,
        )
        cv_results.append(results)
        torch.save(
            results,
            f"{args.save_path}/{model.__class__.__name__}_{args.model_name}_fold_{i}.pt",
        )
        print(f"{args.save_path}/{model.__class__.__name__}_{args.model_name}_fold_{i}.pt")
    torch.save(
        cv_results,
        f"{args.save_path}/{model.__class__.__name__}_{args.model_name}_cv_results_10.pt",
    )
    print(f"stats are saved in {args.save_path}/{model.__class__.__name__}_{args.model_name}_cv_results_10.pt")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='SER_Co_attention', description='train Speech Emotion Recognition Model on IEMOCAP dataset')
    parser.add_argument(
        "--config", type=str, default='./configs/config.yaml', help="configuration .yaml file path"
    )
    parser.add_argument("--batch_size", type=int, default=2, required=False)
    parser.add_argument("--epochs", type=int,default=20)
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--accum_grad", type=int, default=4)
    parser.add_argument(
        "--save_path", default="./checkpoints", help="results file path"
    )
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--cross_val", action="store_true")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--valid_session", type=str, default="Ses01")
    parser.add_argument("--checkpoint", type=str, required=False, default=None)
    parser.add_argument("--num_folds", type=int, default=10, choices=[5,10], help='choose between 10-fold and 5-fold cross validation')
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
        
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model_config = Config(config["COSER"])
    dataset_config = Config(config["DATASET"])
    model_config.lr = args.learning_rate
    model_config.lr_other = float(model_config.lr_other)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    *_, test_dataloader = prepare_dataset_csv(dataset_config, test_rate=0.1)
    model = SER2_transformer_block(40, 512, 512,8, 256)
    if args.cross_val:
        if args.num_folds == 10:
            cross_validation_10_folds(model,(40,512,512,4,256), model_config, dataset_config, args)
        elif args.num_folds == 5:
            cross_validation_5_folds(model,(40,512,512,4,256), model_config, dataset_config, args)
    else:
        train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(
            model_config, dataset_config, args, valid_session=args.valid_session
        )
        print("INITALIZING THE MODEL...")
        gc.collect()
        torch.cuda.empty_cache()
        # model = SER3(40, 512, 512, 4, 256)
        #model = SER_WavLM(40, 512, 512, 4, 256)
        # model = SER_CONV(40, 512,512,1,256,4)
        if args.checkpoint:
            print(f"Model is loaded from checkpoint {args.checkpoint}")
            model.load_state_dict(torch.load(args.checkpoint)["model_state_dict"])
        # model.load_state_dict(torch.load('checkpoints/SER_WavLM_SER_WavLM.pt')['model_state_dict'])

        es = None
        if args.early_stop:
            es = EarlyStopping(patience=7, min_delta=1e-3, type='max')
            print("Early Stopping is activated")

        results = train_run(
            model=model,
            config=model_config,
            epochs=args.epochs,
            train_dl=train_dataloader,
            valid_dl=valid_dataloader,
            batch_size=args.batch_size,
            accum_iter=int(args.accum_grad),
            early_stopping=es,
        )
        torch.save(
            results, f"{args.save_path}/{model.__class__.__name__}_{args.model_name}.pt"
        )
        print("*" * 50)
