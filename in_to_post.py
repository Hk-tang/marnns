
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
# Dyck = DyckLanguage (NUM_PAR, P_VAL, Q_VAL)
# all_letters = word_set = Dyck.return_vocab ()
# n_letters = vocab_size = len (word_set)
#
# print('Loading data...')
#
# training_input, training_output, st = Dyck.training_set_generator (TRAINING_SIZE, MIN_SIZE, MAX_SIZE)
# test_input, test_output, st2 = Dyck.training_set_generator (TEST_SIZE, MAX_SIZE + 2, 2 * MAX_SIZE)
#
# for i in range (1):
#     print (training_output[i])
#     print (Dyck.lineToTensor(training_output[i]))
#     print (Dyck.lineToTensorSigmoid(training_output[i]))

# Number of samples in the training corpus
TRAINING_SIZE = 25000
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



df = pd.read_csv("data/dataset.tsv", sep="\t",header=None)[0:TRAINING_SIZE+TEST_SIZE]
df[0] = df[0].str.replace(" ","")
df[1] = df[1].str.replace(" ","")
# df = df.drop(columns=[2])

def infixToPostfix(infixexpr):
    axns_ohe = []
    axns_str = []

    prec = {}
    prec["*"] = 3
    prec["/"] = 3
    prec["+"] = 2
    prec["-"] = 2
    prec["("] = 1
    opStack = Stack()
    postfixList = []
    tokenList = [char for char in infixexpr]

    for token in tokenList:
#         print(token, end=" ")
        token_vec = [0, 0, 0]  # push, pop, no-op
        token_str = ""
        if token not in prec and token != ")":
            postfixList.append(token)
            token_vec[2] += 1
            token_str += "2"
        elif token == '(':
            opStack.push(token)
            token_vec[0] += 1
            token_str += "0"
        elif token == ')':
            topToken = opStack.pop()
            token_vec[1] += 1
            token_str += "1"
            while topToken != '(':
                postfixList.append(topToken)
                topToken = opStack.pop()
#                 token_vec[1] += 1
        else:
            while (not opStack.isEmpty()) and (prec[opStack.peek()] >= prec[token]):
                postfixList.append(opStack.pop())
                token_vec[1] += 1
            opStack.push(token)
            token_vec[0] += 1
            token_str += "0"
        axns_ohe.append(token_vec)
        axns_str.append(token_str)

    while not opStack.isEmpty():
        postfixList.append(opStack.pop())

    return "".join(axns_str), axns_ohe, "".join(postfixList)

def ohe_axn(df):
    axn_list_list = []
    for item in df.index:
        axn_list_list.append(infixToPostfix(df.iloc[item, 0])[0])

    return axn_list_list

axn_vocab = eqn_vocab = ['(', ')', '*', '+', '-', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# axn_vocab = ["0", "1", "2"]

def lineToTensor(line, n_letters, vocab):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][vocab.index(letter)] = 1.0
    return tensor

lineToTensor("0002021021021", len(axn_vocab), axn_vocab)

lineToTensor("(((9*9)/9)-9)", len(eqn_vocab), eqn_vocab)

lineToTensor("99*9/9-", len(eqn_vocab), eqn_vocab)

# df["axn_list"] = ohe_axn(df)
df["axn_list"] = df[1]

training_input, training_output = df[:TRAINING_SIZE][0].tolist(), df[:TRAINING_SIZE]["axn_list"].tolist()
test_input, test_output = df[TRAINING_SIZE:TRAINING_SIZE+TEST_SIZE][0].tolist(), df[TRAINING_SIZE:TRAINING_SIZE+TEST_SIZE]["axn_list"].tolist()

print(len(training_input), len(test_input))
print(len(training_output), len(test_output))

# Number of hidden units
n_hidden = 256
# Number of hidden layers
n_layers = 1
# Stack size
stack_size = 104
stack_dim = 1

## Stack-RNN with Softmax
# model = SRNN_Softmax (n_hidden, vocab_size, vocab_size, n_layers, stack_size, stack_dim).to(device)
# model = VanillaRNN(n_hidden, vocab_size, vocab_size).to(device)  # works with dyck
model = VanillaRNN(n_hidden, len(axn_vocab), len(eqn_vocab)).to(device)  # works with predicting stack actions
# model = SRNN_Softmax (n_hidden, vocab_size, vocab_size, n_layers, stack_size, stack_dim).to(device)

# Learning rate
learning_rate = .01
# Minimum Squared Error (MSE) loss
criterion = nn.MSELoss()
# criterion = nn.CrossEntropyLoss()
# Adam optimizer (https://arxiv.org/abs/1412.6980)
optim = torch.optim.Adam(model.parameters(), lr = learning_rate)

print ('Model details:')
print (model)

# Number of epochs to train our model
epochs = 2
# Output threshold
epsilon = 0.5

def test_model (model, data_input, data_output, which):
    # Turn on the eval mode
    model.eval()
    # Total number of "correctly" predicted samples
    correct_num = 0
    with torch.no_grad():
        for i in range (len(data_output)):
            len_input = len (data_input[i])
            model.zero_grad ()
            # Initialize the hidden state
            hidden = model.initHidden()
            # Initialize the stack
            stack = torch.zeros (stack_size, stack_dim).to(device)
            # Target values
            if which == "train":
                target = lineToTensor(training_output[i], len(axn_vocab), axn_vocab).to(device)
            else:
                target = lineToTensor(test_output[i], len(axn_vocab), axn_vocab).to(device)
            # Output values
            output_vals = torch.zeros (target.shape)

            for j in range (len(target)):
                if which == "train":
                    output, hidden, stack = model (lineToTensor(training_input[i][j], len(eqn_vocab), eqn_vocab).to(device), hidden, stack)
                else:
                    output, hidden, stack = model (lineToTensor(test_input[i][j], len(eqn_vocab), eqn_vocab).to(device), hidden, stack)
                output_vals [j] = output

            # Binarize the entries based on the output threshold
            out_np = np.int_(output_vals.detach().numpy() >= epsilon)
            target_np = np.int_(target.detach().numpy())

            # (Double-)check whether the output values and the target values are the same
            print(out_np.flatten())
            exit()
            if np.all(np.equal(out_np, target_np)) and (out_np.flatten() == target_np.flatten()).all():
                # If so, increase `correct_num` by one
                correct_num += 1

    return float(correct_num)/len(data_output) * 100, correct_num


def train (model, optimizer, criterion, epoch_num=2):
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
            hidden = model.initHidden()
            # Initialize the stack
            stack = torch.zeros (stack_size, stack_dim).to(device)
            # Target values
            target = lineToTensor(training_output[i], len(axn_vocab), axn_vocab).to(device)
            # Output values
            output_vals = torch.zeros (target.shape)

            for j in range (len(target)):
                output, hidden, stack = model (lineToTensor(training_input[i][j], len(eqn_vocab), eqn_vocab).to(device), hidden, stack)
                output_vals [j] = output
            # MSE (y, y_bar)
            loss = criterion (output_vals, target)
            # Add the current loss to the total loss

            topv, topi = output_vals.topk(1)
            r = topi.squeeze().detach()
            result = [eqn_vocab[i.item()] for i in r]
            result = ''.join(result)
            has_operator = False
            for o in ['+', '-', '*', '/']:
                if o in result:
                    has_operator = True
            if not has_operator:
                loss += 2

            anything = False
            for r in result:
                if r in training_output[i]:
                    anything = True
            if not anything:
                loss += 2
            total_loss += loss.item()
            # Backprop!
            loss.backward ()
            optimizer.step ()

            # Print the performance of the model every 500 steps
            if i % 250 == 0:

                print ('Sample Number {}: '.format(i))
                print ('Input : {}'.format(training_input[i]))
                print ('Expected Output: {}'.format(training_output[i]))
                print ('Output: {}'.format(result))
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

train (model, optim, criterion, epoch_num=epochs)

correct_num = test_model (model, training_input, training_output, "train")
print ('Training accuracy: {}.'.format(correct_num))

correct_num = test_model (model, test_input, test_output, "test")
print ('Test accuracy: {}.'.format(correct_num))

torch.save(model.state_dict(), 'models/vanilla_rnn_model_weights_256.pth')
