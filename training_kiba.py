import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gcn_ori import EmbedDTI_Ori
from models.gcn_pre import EmbedDTI_Pre
from models.gcn_attn import EmbedDTI_Attn
from utils import *

# training function at each epoch
def train(model, device, train_loader_atom, optimizer, epoch):
    print('Training on {} samples ...'.format(len(train_loader_atom.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader_atom):
        data_atom = data[1].to(device)
        data_clique = data[2].to(device)
        optimizer.zero_grad()
        output = model(data_atom,data_clique)
        loss = loss_fn(output, data_atom.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data_atom),
                                                                           len(train_loader_atom),
                                                                           100. * batch_idx / len(train_loader_atom),
                                                                           loss.item()))

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


modeling = [EmbedDTI_Ori,EmbedDTI_Pre,EmbedDTI_Attn][int(sys.argv[1])] # 命令行中传入的第一个参数
model_st = modeling.__name__

# cuda_name = "cuda:0"
# if len(sys.argv)>2:
#     cuda_name = "cuda:" + str(int(sys.argv[2])) 
cuda_name = "cuda:" + str(int(sys.argv[2])) 
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1500

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)


# Main program: iterate over different datasets
print('\nrunning use ', model_st + ' on kiba dataset')
kiba_train_atom = 'data/processed/kiba_train_atom.pt'
kiba_test_atom = 'data/processed/kiba_test_atom.pt'
kiba_train_clique = 'data/processed/kiba_train_clique.pt'
kiba_test_clique = 'data/processed/kiba_test_clique.pt'


if ((not os.path.isfile(kiba_train_atom)) or (not os.path.isfile(kiba_test_atom)) or (not os.path.isfile(kiba_train_clique)) or (not os.path.isfile(kiba_test_clique))):
    print('please run dataloader_kiba.py to prepare data in pytorch format!')
else:
    train_data = TestbedDataset(root='data', dataset='kiba_train')
    test_data = TestbedDataset(root='data', dataset='kiba_test')

    # make data PyTorch mini-batch processing ready
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=True)

    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = modeling().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_mse = 1000
    best_ci = 0
    best_epoch = -1
    model_file_name = 'models/' + model_st + '_kiba.model'
    result = 'result_'+ model_st + '_kiba.csv'
    with open(result,'a') as file:
        file.write('epoch' + ','+ 'mse' + ',' + 'ci' + '\n')
    for epoch in range(NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch+1)
        G,P = predicting(model, device, test_loader)
        ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]
        print('epoch:',epoch+1,'mse:',ret[1],'ci:',ret[-1])
        with open(result,'a') as file:
            file.write(str(epoch+1) + ','+ str(ret[1]) + ',' + str(ret[-1]) + '\n')
        if ret[1]<=best_mse:
            torch.save(model.state_dict(), model_file_name)
            best_epoch = epoch+1
            best_mse = ret[1]
            best_ci = ret[-1]
            print(model_st,'rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci)
        else:
            print(model_st,"current mse: ", ret[1],'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci)
    print('train success!')
