import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MyEnsemble(nn.Module):
    """ Three model ensemble with RNN, LSTM, and GRU pretrained models"""
    def __init__(self, modelA, modelB, modelC, output_dim):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.output = nn.Linear(output_dim*3, output_dim)
    def forward(self, text):
        x1 = self.modelA(text)
        x2 = self.modelB(text)
        x3 = self.modelC(text)
        x = torch.cat((x1, x2, x3), dim=2) #concatenation step 
        final = self.output(x)
        return final

class GRUIOB(nn.Module):
    """GRU model with a post embedding linear layer"""
    def __init__(self, 
                 input_dim, 
                 embedding_dim, 
                 hidden_dim, 
                 output_dim, 
                 n_layers,
                 bidirectional, 
                 dropout, 
                 pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)
        
        self.gru = nn.GRU(embedding_dim, 
                        hidden_dim, 
                        num_layers = n_layers, 
                        bidirectional = bidirectional, 
                        bias = True,
                        dropout = dropout if n_layers > 1 else 0)
        
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        #the times 2 addresses the doubling when it comes to bidirectional models. 
        
        #post embedding fully connected layer
        self.emfc = nn.Linear(embedding_dim, embedding_dim)
        
        #layernorm out of interest 
        self.layer_norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        #text = [sent len, batch size]
        
        embedded = self.dropout(self.embedding(text))
        #embedded = [sent len, batch size, emb dim]
        
        post_emb = self.emfc(embedded) 
        #feed forward connection 
            
        outputs, hidden= self.gru(post_emb)
        #output = [sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hidden dim]
    
        predictions = self.fc(self.dropout(outputs))
        #predictions = [sent len, batch size, output dim]
        #outputs the final layer output for each word of the sentence 
        #does NOT include softmax. That is handled by the criterion algorithm
        
        return predictions

class RNNIOB(nn.Module):
    """ RNN Model class with a relu activation function""" 
    def __init__(self, 
                 input_dim, 
                 embedding_dim, 
                 hidden_dim, 
                 output_dim, 
                 n_layers,
                 bidirectional, 
                 dropout, 
                 pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)
        
        self.rnn = nn.RNN(embedding_dim, 
                        hidden_dim, 
                        num_layers = n_layers, 
                        bidirectional = bidirectional, 
                        bias = True,
                        dropout = dropout if n_layers > 1 else 0, 
                        nonlinearity = 'relu')
        
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        #the times 2 addresses the doubling in 
        
        #post embedding fully connected layer
        self.emfc = nn.Linear(embedding_dim, embedding_dim)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        #text = [sent len, batch size]
        
        embedded = self.dropout(self.embedding(text))
        #embedded = [sent len, batch size, emb dim]

        post_emb = self.emfc(embedded) 
        #feed forward connection post embedding
            
        outputs, hidden= self.rnn(post_emb)
        #output = [sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hidden dim]
    
        predictions = self.fc(self.dropout(outputs))
        #predictions = [sent len, batch size, output dim]
        #outputs the final layer output for each word of the sentence 
        #does NOT include softmax. That is handled by the criterion algorithm
        return predictions

class LSTMIOB(nn.Module):
    """ LSTM Model class """
    def __init__(self, 
                 input_dim, 
                 embedding_dim, 
                 hidden_dim, 
                 output_dim, 
                 n_layers, 
                 bidirectional, 
                 dropout, 
                 pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)
        
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers = n_layers, 
                            bidirectional = bidirectional,
                            dropout = dropout if n_layers > 1 else 0)
        
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        #the times 2 addresses the doubling in 

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        #text = [sent len, batch size]
        
        embedded = self.dropout(self.embedding(text))
        #embedded = [sent len, batch size, emb dim]
        
        outputs, (hidden, cell) = self.lstm(embedded)
        #output = [sent len, batch size, hid dim * n directions]
        #hidden/cell = [n layers * n directions, batch size, hidden dim]
        
        predictions = self.fc(self.dropout(outputs))
        #predictions = [sent len, batch size, output dim]
        
        return predictions