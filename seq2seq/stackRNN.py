## Import relevant libraries and dependencies
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import random

# GPU check
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
#<-------------------Code taken from https://github.com/suzgunmirac/marnns/blob/master/models/rnn_models.py ---------------->
# Taken on 19:11 MDT, March 25, 2021 by Delaney Lothian


## Stack-Augmented RNN with a Softmax Decision Gate
class SRNN_Softmax (nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, n_layers=1, memory_size=104, memory_dim = 3):
        #vocab_size is now input_lang, a Lang object. names should be updated
        super(SRNN_Softmax, self).__init__()
        self.vocab_size = vocab_size.n_words
        self.output_size = output_size.n_words
        self.n_layers = n_layers
        # self.hidden_size = hidden_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.vocab = vocab_size 
        
        self.rnn = nn.RNN(self.vocab_size, self.hidden_size, self.n_layers)# similar to GRU in encoderRNN

        self.W_y = nn.Linear(self.hidden_size, self.output_size)
        self.W_n = nn.Linear(self.hidden_size, self.memory_dim)
        self.W_a = nn.Linear(self.hidden_size, 2)
        self.W_sh = nn.Linear (self.memory_dim, self.hidden_size)
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        
        # Actions -- push : 0 and pop: 1
        self.softmax = nn.Softmax(dim=2) 
        self.sigmoid = nn.Sigmoid ()
    

    def initHidden (self): # named changed from original code
        return torch.zeros (self.n_layers, 1, self.hidden_size).to(device)
    

    def initStack(self): #function added
        return torch.zeros (self.memory_size, self.memory_dim).to(device)

    def to_one_hot_vector(self, inp):
        # print("self.vocab.index2word", self.vocab.index2word)
        # target = self.vocab.index2word[int(inp.item())]
        y = torch.zeros(self.vocab_size)
        y[inp]=1
        y = y.view(1, 1, -1)
        return y
        

    def forward(self, inp, hidden0, stack, temperature=1.):
        # embedded = self.embedding(inp).view(1, 1, -1)
        inp = self.to_one_hot_vector(inp)
        temp = self.W_sh (stack[0]).view(1, 1, -1)
        hidden_bar = self.W_sh (stack[0]).view(1, 1, -1) + hidden0
        # print("inp", inp.shape)
        # print("embedded", embedded)
        ht, hidden = self.rnn(inp, hidden_bar)
        # ht, hidden = self.rnn(embedded, hidden_bar)
        output = self.sigmoid(self.W_y(ht)).view(-1, self.output_size)
        self.action_weights = self.softmax (self.W_a (ht)).view(-1)
        self.new_elt = self.sigmoid (self.W_n(ht)).view(1, self.memory_dim)
        push_side = torch.cat ((self.new_elt, stack[:-1]), dim=0)
        pop_side = torch.cat ((stack[1:], torch.zeros(1, self.memory_dim).to(device)), dim=0)
        stack = self.action_weights [0] * push_side + self.action_weights [1] * pop_side

        return output, hidden, stack


# <---------------------------------------------------------------------------------->