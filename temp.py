# Import libraries and relevant dependencies
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import string
from torch.autograd import Variable

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

# GPU/CPU Check
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") ## GPU stuff
print (device)


## Parameters of the Probabilistic Dyck Language 
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
    print (training_input[i])
    print (training_output[i])
    print (Dyck.lineToTensor(training_input[i]))
    print (Dyck.lineToTensorSigmoid(training_output[i]))
    print (test_input[i])
    print (test_output[i])


# Number of hidden units
n_hidden = 8
# Number of hidden layers
n_layers = 1
# Stack size
stack_size = 104
stack_dim = 1

# Parameters for the temperature-based methods
temp = 1.0
temp_min = 0.5
ANNEAL_RATE = .00001

## Stack-RNN with Softmax
model = SRNN_Softmax (n_hidden, vocab_size, vocab_size, n_layers, stack_size, stack_dim).to(device)
# Learning rate
learning_rate = .01
# Minimum Squared Error (MSE) loss
criterion = nn.MSELoss() 
# Adam optimizer (https://arxiv.org/abs/1412.6980)
optim = torch.optim.Adam(model.parameters(), lr = learning_rate)

print ('Model details:')
print (model)


# Number of epochs to train our model
epoch_num = 3
# Output threshold
epsilon = 0.5

def test_model (model, data_input, data_output):
    # Turn on the eval mode
    model.eval()
    # Total number of "correctly" predicted samples
    correct_num = 0
    with torch.no_grad():
        for i in range (len(data_output)):
            len_input = len (data_input[i])
            model.zero_grad ()
            # Initialize the hidden state
            hidden = model.init_hidden()
            # Initialize the stack
            stack = torch.zeros (stack_size, stack_dim).to(device)
            # Target values
            target = Dyck.lineToTensorSigmoid(data_output[i]).to(device) 
            # Output values
            output_vals = torch.zeros (target.shape)
            
            for j in range (len_input):
                output, hidden, stack = model (Dyck.lineToTensor(data_input[i][j]).to(device), hidden, stack)
                output_vals [j] = output

            # Binarize the entries based on the output threshold
            out_np = np.int_(output_vals.detach().numpy() >= epsilon)
            target_np = np.int_(target.detach().numpy())
            
            # (Double-)check whether the output values and the target values are the same
            if np.all(np.equal(out_np, target_np)) and (out_np.flatten() == target_np.flatten()).all():
                # If so, increase `correct_num` by one
                correct_num += 1
                
    return float(correct_num)/len(data_output) * 100, correct_num


def train (model, optimizer, criterion, epoch_num=5):
    # Turn on the train model for the model
    model.train()
    # Arrays for loss and "moving" accuracy per epoch
    loss_arr = []
    correct_arr = []
    for epoch in range(1, epoch_num + 1):
        print ('Epoch: {}'.format(epoch))
        
        # Total loss per epoch
        total_loss = 0
        # Total number of "correctly" predicted samples so far in the epoch
        counter = 0

        for i in range (TRAINING_SIZE):
            len_input = len (training_input[i])
            # Good-old zero grad
            model.zero_grad ()
            # Initialize the hidden state
            hidden = model.init_hidden()
            # Initialize the stack 
            stack = torch.zeros (stack_size, stack_dim).to(device)
            # Target values
            target = Dyck.lineToTensorSigmoid(training_output[i]).to(device) 
            # Output values
            output_vals = torch.zeros (target.shape)

            for j in range (len_input):
                output, hidden, stack = model (Dyck.lineToTensor(training_input[i][j]).to(device), hidden, stack)
                output_vals [j] = output
                exit()
            
            # MSE (y, y_bar)
            loss = criterion (output_vals, target)
            # Add the current loss to the total loss
            total_loss += loss.item()
            # Backprop! 
            loss.backward ()
            optimizer.step ()
            
            # Print the performance of the model every 500 steps
            if i % 250 == 0:
                print ('Sample Number {}: '.format(i))
                print ('Input : {}'.format(training_input[i]))
                print ('Output: {}'.format(training_output[i]))
                print ('* Counter: {}'.format(counter))
                print ('* Avg Loss: {}'.format(total_loss/(i+1))) 

            # Binarize the entries based on the output threshold
            out_np = np.int_(output_vals.detach().numpy() >= epsilon)
            target_np = np.int_(target.detach().numpy())
                
            # "Moving" training accuracy
            if np.all(np.equal(out_np, target_np)) and (out_np.flatten() == target_np.flatten()).all():
                counter += 1
                
            # At the end of the epoch, append our total loss and "moving" accuracy
            if i == TRAINING_SIZE - 1:
                print ('Counter: {}'.format(float(counter)/TRAINING_SIZE))
                loss_arr.append (total_loss)
                correct_arr.append(counter) 

        if epoch % 1 == 0:
            print ('Training Accuracy %: ', correct_arr)
            print ('Loss: ', loss_arr)

train (model, optim, criterion)