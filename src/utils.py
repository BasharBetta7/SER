def evaluate_metrics(ypred, ytrue):
    # ypred : (B), ytrue : (B)
    ypred = torch.tensor(ypred, dtype=torch.float32)
    ytrue = torch.tensor(ytrue, dtype=torch.float32)
    ua = (ypred == ytrue).float().mean()
    ypred_onehot = torch.eye(4)[ypred.long()]
    ytrue_onehot = torch.eye(4)[ytrue.long()]
    wa = torch.mean(torch.sum((ypred_onehot == ytrue_onehot)*ytrue_onehot, 0)/torch.sum(ytrue_onehot.int(),0))
    return wa.item(), ua.item()

yp = [0,0,0,1,2,1,2,0,3]
yt = [0,0,0,0,1,2,2,3,3]
evaluate_metrics(yp, yt)

# preprocess audio for wav2vec compatitblity 
def create_processor(wav2vec_config):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_config)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(wav2vec_config, do_lower_case=False,  word_delimiter_token="|")
    return Wav2Vec2Processor(feature_extractor, tokenizer)

audio_processor = create_processor(wav2vec_config)


