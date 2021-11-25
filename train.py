import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.legacy import data, datasets
import copy 
from tqdm import tqdm
import spacy
import time
import random 

#importing custom helper functions
from helpers import *
from config import * # loads data and all variables needed
from model import *

#Seed setting over all possible random components.
seed_everything() 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token] 

############################################################
# DATA PREPROCESSING IS DONE DURING THE CONFIG LOADING PHASE
############################################################

##########################################################
# DEFINE ITERATOR
########################################################## 

train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, valid_data), 
    batch_size = BATCH_SIZE,
    device = device) 

##########################################################
# GRU TRAINING 
##########################################################

model = GRUIOB(INPUT_DIM, 
                EMBEDDING_DIM, 
                HIDDEN_DIM, 
                OUTPUT_DIM, 
                N_LAYERS, 
                BIDIRECTIONAL, 
                DROPOUT, 
                PAD_IDX)

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

initialiser(model)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

#optimiser 
optimizer = optim.Adam(model.parameters()) 

#ensuring that the pad tokens are ignored during the cross entropy calculation 
TAG_PAD_IDX = LABELS.vocab.stoi[LABELS.pad_token]

#criterion 
criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX)

#sends the models to the GPU 
model = model.to(device)
criterion = criterion.to(device)

#import necessary training functions
from helpers import train, evaluate, epoch_time, count_parameters

#N_EPOCHS IS STORED IN THE CONFIG FILE
best_valid_loss = float('inf')
start_time = time.time() 
loop =  tqdm(range(N_EPOCHS),leave=False)
for epoch in loop:
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, TAG_PAD_IDX)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, TAG_PAD_IDX)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')

    loop.set_description(f"Epoch [{epoch+1}/{N_EPOCHS}]")
    loop.set_postfix(train_loss=train_loss, valid_loss=valid_loss)

end_time = time.time()
epoch_mins, epoch_secs = epoch_time(start_time, end_time)
print("==============================================================================")
print("GRU Model Performance:")
print(f'Total Epochs: {epoch+1:02}   | Totals Time: {epoch_mins}m {epoch_secs}s')
print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
print(f'Val. Loss: {valid_loss:.3f}  |  Val. Acc: {valid_acc*100:.2f}%')
print(f'The model has {count_parameters(model):,} trainable parameters')
print("==============================================================================")
#Model assignment for ensemble  
model_GRU = copy.deepcopy(model)

##########################################################
# RNN TRAINING 
##########################################################

model = RNNIOB(INPUT_DIM, 
                EMBEDDING_DIM, 
                HIDDEN_DIM, 
                OUTPUT_DIM, 
                N_LAYERS, 
                BIDIRECTIONAL, 
                DROPOUT, 
                PAD_IDX)

#intialiser
initialiser(model)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

#optimiser 
optimizer = optim.Adam(model.parameters()) 

#ensuring that the pad tokens are ignored during the cross entropy calculation 
TAG_PAD_IDX = LABELS.vocab.stoi[LABELS.pad_token]

#criterion 
criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX)

#sends the models to the GPU 
model = model.to(device)
criterion = criterion.to(device)

#main training lopp 
best_valid_loss = float('inf')
start_time = time.time() 
loop =  tqdm(range(N_EPOCHS),leave=False)
for epoch in loop:
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, TAG_PAD_IDX)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, TAG_PAD_IDX)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut2-model.pt')

    loop.set_description(f"Epoch [{epoch+1}/{N_EPOCHS}]")
    loop.set_postfix(train_loss=train_loss, valid_loss=valid_loss)
    
end_time = time.time()
epoch_mins, epoch_secs = epoch_time(start_time, end_time)
print("RNN Model Performance")
print(f'Total Epochs: {epoch+1:02}  | Totals Time: {epoch_mins}m {epoch_secs}s')
print(f'Train Loss: {train_loss:.3f}| Train Acc: {train_acc*100:.2f}%')
print(f'Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
print(f'The model has {count_parameters(model):,} trainable parameters') 
print("==============================================================================")
#Model assignment for ensemble  
model_RNN = copy.deepcopy(model) 

##########################################################
# LSTM TRAINING 
##########################################################

model = LSTMIOB(INPUT_DIM, 
                EMBEDDING_DIM, 
                HIDDEN_DIM, 
                OUTPUT_DIM, 
                N_LAYERS, 
                BIDIRECTIONAL, 
                DROPOUT, 
                PAD_IDX)

##intialiser
initialiser(model)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
#optimiser 
optimizer = optim.Adam(model.parameters()) 

#ensuring that the pad tokens are ignored during the cross entropy calculation 
TAG_PAD_IDX = LABELS.vocab.stoi[LABELS.pad_token]

#criterion 
criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX)

#sends the models to the GPU 
model = model.to(device)
criterion = criterion.to(device)

#main training loop
best_valid_loss = float('inf')
start_time = time.time() 
loop =  tqdm(range(N_EPOCHS),leave=False)
for epoch in loop:
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, TAG_PAD_IDX)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, TAG_PAD_IDX)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut3-model.pt')

    loop.set_description(f"Epoch [{epoch+1}/{N_EPOCHS}]")
    loop.set_postfix(train_loss=train_loss, valid_loss=valid_loss)

end_time = time.time()
epoch_mins, epoch_secs = epoch_time(start_time, end_time)
print("LSTM Model Performance")
print(f'Total Epochs: {epoch+1:02}  | Totals Time: {epoch_mins}m {epoch_secs}s')
print(f'Train Loss: {train_loss:.3f}| Train Acc: {train_acc*100:.2f}%')
print(f'Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
print(f'The model has {count_parameters(model):,} trainable parameters')
print("==============================================================================")
#Model assignment for ensemble 
model_LSTM = copy.deepcopy(model) 

##########################################################
# ENSEMBLE TRAINING 
##########################################################

#freeze models 
for param in model_GRU.parameters():
    param.requires_grad = False

for param in model_RNN.parameters():
    param.requires_grad = False

for param in model_LSTM.parameters():
    param.requires_grad = False


#intialise model 
model = MyEnsemble(model_GRU, model_RNN, model_LSTM ,OUTPUT_DIM)

#optimiser 
optimizer = optim.Adam(model.parameters()) 

#ensuring that the pad tokens are ignored during the cross entropy calculation 
TAG_PAD_IDX = LABELS.vocab.stoi[LABELS.pad_token]

#criterion 
criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX)

#sends the models to the GPU 
model = model.to(device)
criterion = criterion.to(device)

best_valid_loss = float('inf')
start_time = time.time() 
loop =  tqdm(range(N_EPOCHS),leave=False)
for epoch in loop:
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, TAG_PAD_IDX)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, TAG_PAD_IDX)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut4-model.pt')

    loop.set_description(f"Epoch [{epoch+1}/{N_EPOCHS}]")
    loop.set_postfix(train_loss=train_loss, valid_loss=valid_loss)

end_time = time.time()
epoch_mins, epoch_secs = epoch_time(start_time, end_time)
print("ENSEMBLE PRETRAINED GRU + RNN + LSTM Model Performance:")
print(f'Total Epochs: {epoch+1:02}  | Totals Time: {epoch_mins}m {epoch_secs}s')
print(f'Train Loss: {train_loss:.3f}| Train Acc: {train_acc*100:.2f}%')
print(f'Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
print(f'The model has {count_parameters(model):,} trainable parameters')
print("==============================================================================")
