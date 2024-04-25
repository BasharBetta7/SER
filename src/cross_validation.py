from imports import *
from dataset import IEMOCAP_Dataset
from prepare_dataset import prepare_dataset_csv
from run import collate_2

# 5-fold cross validation 
def cross_validation(config):
    cv_results = []
    print('CROSS VALIDATION')
    for i in range(1,6):
        valid_session   = 'Ses0'+ str(i)
        train,val,test  = prepare_dataset_csv(valid_session)
        train_dataset   = IEMOCAP_Dataset(train)
        valid_dataset   = IEMOCAP_Dataset(val)
        test_dataset    = IEMOCAP_Dataset(test)
    
        #create dataloader for train:
        train_dataloader = DataLoader(dataset=train_dataset, 
                                      batch_size = 4,
                                      shuffle=True,
                                      num_workers= 1,
                                      collate_fn= collate_2)
    
        # and for validation    
        valid_dataloader = DataLoader(dataset= valid_dataset,
                                      batch_size= 4,
                                      num_workers= 1,
                                  shuffle= False,
                                  collate_fn= collate_2)
    
        # re-initialize the model
        
       
        
        print('Re-initialize the model...')
        config.model.to('cpu')
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2)
        config.model.__init__(40, 256, 256, 1, 256)
        config.model.to(device)
        config.train_dataloader = train_dataloader
        config.valid_dataloader = valid_dataloader
        try:
            results = train_run(config)
            cv_results.append(results)
        except :
            print('CUDA OOM at fold ')
       
    return cv_results
    # here we make comparisons for the loss: 
