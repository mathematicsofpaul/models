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
from config import * 
from model import *

#Seed setting over all possible random components.
seed_everything()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############################################################
# DATA PREPROCESSING IS DONE DURING THE CONFIG LOADING PHASE
############################################################

########################################################
#RELOADING ALL WEIGHTS
########################################################

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
#model_GRU.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

model_RNN = RNNIOB(INPUT_DIM, 
                EMBEDDING_DIM, 
                HIDDEN_DIM, 
                OUTPUT_DIM, 
                N_LAYERS, 
                BIDIRECTIONAL, 
                DROPOUT, 
                PAD_IDX)



model_RNN.load_state_dict(torch.load('tut2-model.pt'))
#model_RNN.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

model_LSTM = LSTMIOB(INPUT_DIM, 
                EMBEDDING_DIM, 
                HIDDEN_DIM, 
                OUTPUT_DIM, 
                N_LAYERS, 
                BIDIRECTIONAL, 
                DROPOUT, 
                PAD_IDX)

model_LSTM.load_state_dict(torch.load('tut3-model.pt'))
#model_LSTM.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

model = MyEnsemble(model_GRU, model_RNN, model_LSTM ,OUTPUT_DIM)

#ensuring that the pad tokens are ignored during the cross entropy calculation 
TAG_PAD_IDX = LABELS.vocab.stoi[LABELS.pad_token]

#sends the models to the GPU 
model = model.to(device)
model.load_state_dict(torch.load('tut4-model.pt'))

import sys 

sentence = sys.argv[1]

tokens, tags, unks = tag_sentence(model, 
                                  device, 
                                  sentence, 
                                  TEXT, 
                                  LABELS) 
print(tags)

#TRY THE FOLLOWING!
#python pipeline.py "Clustering of missense mutations in the ataxia-telangiectasia gene in a sporadic T-cell leukaemia." 
