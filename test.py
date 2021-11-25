import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy 
from tqdm import tqdm
from torchtext.legacy import data, datasets

import spacy
import numpy as np

import time
import random 

#importing custom helper functions
from helpers import *
from model import * 
from config import * #all preprocessed data is loaded through config launch

#Seed setting over all possible random components.
seed_everything()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token] 

##########################################################
# DEFINE ITERATOR
########################################################## 
#kept here for structurals sake. 

train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, valid_data), 
    batch_size = BATCH_SIZE,
    device = device) 

########################################################
#RELOADING ALL WEIGHTS & DEFINING MODELS
########################################################

#define model intializer 
def initialiser(model): 
    #normal initialisation 
    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.normal_(param.data, mean = 0, std = 0.1)

    #applying the normal weights        
    model.apply(init_weights)

    #embeddding initialisation
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings) 

    #pad token embedding to zero vector
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

#Define GRU of Ensemble 
model_GRU = GRUIOB(INPUT_DIM, 
                EMBEDDING_DIM, 
                HIDDEN_DIM, 
                OUTPUT_DIM, 
                N_LAYERS, 
                BIDIRECTIONAL, 
                DROPOUT, 
                PAD_IDX) 

model_GRU.load_state_dict(torch.load('tut1-model.pt'))

model_RNN = RNNIOB(INPUT_DIM, 
                EMBEDDING_DIM, 
                HIDDEN_DIM, 
                OUTPUT_DIM, 
                N_LAYERS, 
                BIDIRECTIONAL, 
                DROPOUT, 
                PAD_IDX)

model_RNN.load_state_dict(torch.load('tut2-model.pt'))

model_LSTM = LSTMIOB(INPUT_DIM, 
                EMBEDDING_DIM, 
                HIDDEN_DIM, 
                OUTPUT_DIM, 
                N_LAYERS, 
                BIDIRECTIONAL, 
                DROPOUT, 
                PAD_IDX)

model_LSTM.load_state_dict(torch.load('tut3-model.pt'))

########################################################
#ENSEMBLE TESTING
########################################################

model = MyEnsemble(model_GRU, model_RNN, model_LSTM ,OUTPUT_DIM)

#ensuring that the pad tokens are ignored during the cross entropy calculation 
TAG_PAD_IDX = LABELS.vocab.stoi[LABELS.pad_token]

criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX)
#sends the models to the GPU 
model = model.to(device)
criterion = criterion.to(device)
model.load_state_dict(torch.load('tut4-model.pt'))

test_loss, test_acc = evaluate(model, valid_iterator, criterion, TAG_PAD_IDX)
print("ENSEMBLE PRETRAINED GRU + RNN + LSTM Model Performance:")
print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%') 
