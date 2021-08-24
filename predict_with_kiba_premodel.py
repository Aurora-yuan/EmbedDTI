import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gcn_pre import GCNNet_Pre
from utils import *


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples ...'.format(len(loader.dataset)))
    with torch.no_grad():
        for idx,data in enumerate(loader):
            data_atom = data[1].to(device)
            data_clique = data[2].to(device)
        
            output = model(data_atom,data_clique)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_atom.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


modeling = [GCNNet_Pre][int(sys.argv[1])]
cuda_name = "cuda:1"

TEST_BATCH_SIZE = 512



result = []
name_atom = 'data/processed/kiba_test_atom.pt'
name_clique = 'data/processed/kiba_test_clique.pt'


if ((not os.path.isfile(name_atom)) or (not os.path.isfile(name_clique))):
    print('please run dataloader_kiba.py to prepare data in pytorch format!')
else:
    test_data = TestbedDataset(root='data', dataset='kiba_test')
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=True)
    model_st = modeling.__name__
    print('\npredicting for testdataset using ', model_st)
    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = modeling().to(device)
    model_file_name = 'models/model_GCNNet_kiba_pre.model'
    if os.path.isfile(model_file_name):            
        model.load_state_dict(torch.load(model_file_name))
        G,P = predicting(model, device, test_loader)
        ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]
        ret = ['kiba',model_st]+[round(e,3) for e in ret]
        result += [ ret ]
        print('dataset,model,rmse,mse,pearson,spearman,ci')
        print(ret)
    else:
        print('model is not available!')



with open('results/result_kiba_pre.csv','w') as f:
    f.write('dataset,model,rmse,mse,pearson,spearman,ci\n')
    for ret in result:
        f.write(','.join(map(str,ret)) + '\n')


