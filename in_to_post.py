# Import libraries and relevant dependencies
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import string
from torch.autograd import Variable
import pandas as pd

# Dyck library
from tasks.dyck_generator import DyckLanguage

# RNN Models
from models.rnn_models import VanillaRNN, SRNN_Softmax, SRNN_Softmax_Temperature, SRNN_GumbelSoftmax

# Set default tensor type "double"
torch.set_default_tensor_type('torch.DoubleTensor')

randomseed_num = 23
print ('RANDOM SEED: {}'.format(randomseed_num))
random.seed (randomseed_num)
np.random.seed (randomseed_num)
torch.manual_seed(randomseed_num)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") ## GPU stuff
print (device)

NUM_PAR = 2
MIN_SIZE = 2
MAX_SIZE = 50
P_VAL = 0.5
Q_VAL = 0.25

# Number of samples in the training corpus
TRAINING_SIZE = 5000
# Number of samples in the test corpus
TEST_SIZE = 5000

# Create a Dyck language generator
Dyck = DyckLanguage (NUM_PAR, P_VAL, Q_VAL)
all_letters = word_set = Dyck.return_vocab ()
n_letters = vocab_size = len (word_set)

print('Loading data...')

training_input, training_output, st = Dyck.training_set_generator (TRAINING_SIZE, MIN_SIZE, MAX_SIZE)
test_input, test_output, st2 = Dyck.training_set_generator (TEST_SIZE, MAX_SIZE + 2, 2 * MAX_SIZE)

for i in range (1):
    print (training_output[i])
    print (Dyck.lineToTensor(training_output[i]))
    print (Dyck.lineToTensorSigmoid(training_output[i]))

# Number of samples in the training corpus
TRAINING_SIZE = 5000
# Number of samples in the test corpus
TEST_SIZE = 5000

class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def update(self, item):
        self.items[len(self.items) - 1] = item
        return self.items

    def peek(self):
        return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)

    

df = pd.read_csv("dataset.tsv.txt", sep="\t",header=None)[0:TRAINING_SIZE+TEST_SIZE]
df[0] = df[0].str.replace(" ","") 
df[1] = df[1].str.replace(" ","")
df = df.drop(columns=[2])