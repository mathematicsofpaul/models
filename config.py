import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.legacy import data, datasets
import copy 
from tqdm import tqdm
import spacy
from helpers import *

#RELATIVE PATHS
seed_everything() 

TRAIN_PATH = './data/NERdata/train.tsv' 
TEST_PATH = './data/NERdata/test.tsv' 

###############################################
#ALL DATA PREPROCESSING IS DONE HERE
###############################################

TEXT = data.Field(lower = False) #depending on case sensative or not 
LABELS = data.Field(unk_token = None)

train_data = datasets.SequenceTaggingDataset(
                path=TRAIN_PATH,
                fields=[('text', TEXT),
                        ('labels', LABELS)])   

valid_data = datasets.SequenceTaggingDataset(
                path=TEST_PATH,
                fields=[('text',  TEXT),
                        ('labels', LABELS)])

fields = (("text", TEXT), ("labels", LABELS)) 

#builds a vocab library for the training data. 
TEXT.build_vocab(train_data, 
                 vectors = "glove.6B.300d",
                 unk_init = torch.Tensor.normal_)

#builds a vocab library for the labels 
LABELS.build_vocab(train_data)

###############################################
#RNN VALUES 
###############################################

#ALL VARIABLES FOR RNN's 
INPUT_DIM = len(TEXT.vocab)
OUTPUT_DIM = len(LABELS.vocab)
BATCH_SIZE = 128 
EMBEDDING_DIM = 300
HIDDEN_DIM = 32
N_LAYERS = 2 #number of stacked layers 2 to 3 is pretty good 
BIDIRECTIONAL = True #we elect to use a bidirectional version 
DROPOUT = 0.25
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token] 

#NUMBER OF EPOCHS 
N_EPOCHS = 40