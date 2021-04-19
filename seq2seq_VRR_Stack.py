import time
import random
import math
import csv
import unicodedata
import re
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from models.rnn_models import VanillaRNN
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import string
from torch.autograd import Variable
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

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
TRAINING_SIZE = 30000
# Number of samples in the test corpus
TEST_SIZE = 0

# Create a Dyck language generator
Dyck = DyckLanguage (NUM_PAR, P_VAL, Q_VAL)
all_letters = word_set = Dyck.return_vocab ()
n_letters = vocab_size = len (word_set)

# print('Loading data...')

# training_input, training_output, st = Dyck.training_set_generator (TRAINING_SIZE, MIN_SIZE, MAX_SIZE)
# test_input, test_output, st2 = Dyck.training_set_generator (TEST_SIZE, MAX_SIZE + 2, 2 * MAX_SIZE)

# for i in range (1):
#     print (training_output[i])
#     print (Dyck.lineToTensor(training_output[i]))
#     print (Dyck.lineToTensorSigmoid(training_output[i]))

# Number of samples in the training corpus
TRAINING_SIZE = 30000
# Number of samples in the test corpus
TEST_SIZE = 0

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

    

df = pd.read_csv("data/dataset_len_5_10.tsv", sep="\t",header=None)[0:TRAINING_SIZE+TEST_SIZE]
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

eqn_vocab = ['(', ')', '*', '+', '-', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
axn_vocab = ["0", "1", "2"]

def lineToTensor(line, n_letters, vocab):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][vocab.index(letter)] = 1.0
    return tensor

lineToTensor("0002021021021", len(axn_vocab), axn_vocab)

lineToTensor("(((9*9)/9)-9)", len(eqn_vocab), eqn_vocab)

df["axn_list"] = ohe_axn(df)

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
# Adam optimizer (https://arxiv.org/abs/1412.6980)
optim = torch.optim.Adam(model.parameters(), lr = learning_rate)

print ('Model details:')
print (model)

# Number of epochs to train our model
epochs = 2
# Output threshold
epsilon = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 100

teacher_forcing_ratio = 0.5


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "?", 1: "!"}
        # self.index2word = {}
        self.n_words = 2  # Count SOS and EOS
        # self.n_words = 0

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")
    pairs = []
    with open("data/dataset_len_5_10.tsv", encoding='utf-8') as tsv:
        reader = csv.reader(tsv, delimiter="\t")
        for line in reader:
            pairs.append([line[0], line[1]])

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


## Good-Old Vanilla RNN Model
class VanillaRNN_mod(nn.Module):
    def __init__(self, hidden_dim, output_size, vocab_size, n_layers=1,
                 memory_size=1, memory_dim=1):
        super(VanillaRNN_mod, self).__init__()
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_size = hidden_dim
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.rnn = nn.RNN(self.vocab_size, self.hidden_size, self.n_layers)
        # self.rnn = nn.GRU(self.vocab_size, self.hidden_size, self.n_layers)
        self.W_y = nn.Linear(self.hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.embedding = nn.Embedding(self.hidden_size, self.vocab_size)

    def initHidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size).to(device)

    def forward(self, input, hidden0, stack=None, temperature=1.):
        try:
            embedded = self.embedding(input).view(1, 1, -1)
        except:
            print(input.item())
            exit()
        ht, hidden = self.rnn(embedded, hidden0)
        output = self.sigmoid(self.W_y(ht)).view(-1, self.output_size)
        return output, hidden  # , stack


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


## Stack-Augmented RNN with a Softmax Decision Gate
class SRNN_Softmax(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, n_layers=1,
                 memory_size=104, memory_dim=3):
        # vocab_size is now input_lang, a Lang object. names should be updated
        super(SRNN_Softmax, self).__init__()
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.n_layers = n_layers
        # self.hidden_size = hidden_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        # self.embedding = nn.Embedding(self.output_size, hidden_size)

        # self.rnn = nn.RNN(self.vocab_size, self.hidden_size, self.n_layers)# similar to GRU in encoderRNN
        self.rnn = nn.GRU(self.vocab_size, self.hidden_size, self.n_layers)

        self.W_y = nn.Linear(self.hidden_size, self.output_size)
        self.W_n = nn.Linear(self.hidden_size, self.memory_dim)
        self.W_a = nn.Linear(self.hidden_size, 2)
        self.W_sh = nn.Linear(self.memory_dim, self.hidden_size)
        self.embedding = nn.Embedding(self.hidden_size, self.vocab_size)

        # Actions -- push : 0 and pop: 1
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

    def initHidden(self):  # named changed from original code
        return torch.zeros(self.n_layers, 1, self.hidden_size).to(device)

    def initStack(self):  # function added
        return torch.zeros(self.memory_size, self.memory_dim).to(device)

    def to_one_hot_vector(self, inp):
        # print("self.vocab.index2word", self.vocab.index2word)
        # target = self.vocab.index2word[int(inp.item())]
        y = torch.zeros(self.vocab_size)
        y[inp] = 1
        y = y.view(1, 1, -1)
        return y

    def forward(self, inp, hidden0, stack, temperature=1.):
        embedded = self.embedding(inp).view(1, 1, -1)

        hidden_bar = self.W_sh(stack[0]).view(1, 1, -1) + hidden0

        ht, hidden = self.rnn(embedded, hidden_bar)
        output = self.sigmoid(self.W_y(ht)).view(-1, self.output_size)

        self.action_weights = self.softmax(self.W_a(ht)).view(-1)
        
        self.new_elt = self.sigmoid(self.W_n(ht)).view(1, self.memory_dim)
        push_side = torch.cat((self.new_elt, stack[:-1]), dim=0)
        pop_side = torch.cat(
            (stack[1:], torch.zeros(1, self.memory_dim).to(device)), dim=0)
        stack = self.action_weights[0] * push_side + self.action_weights[
            1] * pop_side

        return output, hidden, stack


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1,
                 max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def train(input_tensor, target_tensor, stack_output_tensor, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    encoder_stack = encoder.initStack()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    stack_target_length = stack_output_tensor.size(0)
    
    output_vals = torch.zeros (input_length)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size,
                                  device=device)

    loss = 0
    criterion2 = nn.MSELoss() 
    # print("stack_output_tensor.shape", stack_output_tensor.shape)
    # print("input_tensor.shape", input_tensor.shape)
    for ei in range(input_length):
        # encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                #  encoder_hidden)
        encoder_output, encoder_hidden, encoder_stack = encoder(input_tensor[ei], encoder_hidden, encoder_stack)
        encoder_outputs[ei] = encoder_output[0, 0]
        # print("encoder_output", encoder_output)
        # print("encoder_output.shape", encoder_output.shape)
        # print("stack_output_tensor[ei]", stack_output_tensor[ei])
        # print("stack_output_tensor[ei].shape", stack_output_tensor[ei].shape)
        # loss += criterion2 (encoder_output, stack_output_tensor[ei]) # UNCOMMENT THIS LINE


    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    answer = ''
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
            topv, topi = decoder_output.data.topk(1)
            answer += output_lang.index2word[topi.item()]

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            if di != 0:
                answer += output_lang.index2word[topi.item()]
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])

            if decoder_input.item() == EOS_token:
                break

    target = [output_lang.index2word[i.item()] for i in target_tensor]
    loss.backward()

    bleu_score = corpus_bleu([target], [list(answer)], weights=(1, 0, 0, 0))

    target = ''.join(target)
    # if '!' == target[-1]:
    #     target = target[:-1]
    ind_acc = 0

    # if '!' == target[-1]:
    #     target = target[:-1]
    # if '?' in answer:
    #     target = target[:target.find('?')]
    # if '!' == answer[-1]:
    #     answer = answer[:-1]

    acc = 1 if target == answer else 0

    if len(answer) < len(target):
        l = len(answer)
    else:
        l = len(target)
    for i in range(l):
        if target[i] == answer[i]:
            ind_acc += 1

    ind_acc /= len(target)
    results = [target, answer]

    encoder_optimizer.step()
    decoder_optimizer.step()
    l = loss.item() / target_length
    return l, acc, bleu_score, ind_acc, results  # , encoder_stack


def indexesFromSentence(lang, sentence):
    indices = []
    for word in sentence:
        indices.append(lang.word2index[word])
    return indices


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100,
               learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    total_acc = 0
    bleu = 0
    ind_acc = 0
    axn_vocab = ["0", "1", "2", "!"]

    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)
    # training_pairs = [tensorsFromPair(random.choice(pairs))
    #                   for i in range(n_iters)]
    training_pairs = [tensorsFromPair(pairs[i])
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    ind_accs = []

    for iter in range(1, n_iters + 1):
        # print("pairs[iter-1]", pairs[iter-1])
        # print("len(pairs[iter-1][0])", len(pairs[iter-1][0]))

        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        stack_output = training_output[iter - 1]
        stack_output += '!'
        # print("len(stack_output)", len(stack_output))
        stack_output_tensor = lineToTensor(stack_output,  len(axn_vocab), axn_vocab)
        loss, acc, bleu_score, i_acc, results = train(input_tensor,
                                                      target_tensor, stack_output_tensor, encoder,
                                                      decoder,
                                                      encoder_optimizer,
                                                      decoder_optimizer,
                                                      criterion)
        print_loss_total += loss
        plot_loss_total += loss
        total_acc += acc
        bleu += bleu_score
        ind_acc += i_acc

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_bleu_avg = bleu / print_every
            print_ind_acc = ind_acc / print_every
            print_run_acc = total_acc / print_every
            print_loss_total = 0
            bleu = 0
            ind_acc = 0
            total_acc = 0
            print("ex: actual:", results[0], '\npredicted:', results[1])
            print(
                '%s (%d %d%%) avg loss: %.4f \navg bleu: %.4f \nrunning acc: %.4f \nind acc avg: %.4f' % (
                timeSince(start, iter / n_iters),
                iter, iter / n_iters * 100,
                print_loss_avg, print_bleu_avg, print_run_acc, print_ind_acc))
            ind_accs.append(print_ind_acc)

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    # showPlot(plot_losses)
    return print_loss_avg, print_run_acc, ind_accs, print_bleu_avg


def evaluate(encoder, decoder, sentence, input_lang, max_length):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        # encoder_stack = encoder.initStack()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size,
                                      device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            # encoder_output, encoder_hidden, encoder_stack = encoder(input_tensor[ei], encoder_hidden, encoder_stack)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


if __name__ == "__main__":
    input_lang, output_lang, pairs = prepareData("infix", 'postfix')
    val_size = round(0.1 * len(pairs))
    val_pairs = pairs[len(pairs) - val_size:]
    pairs = pairs[:len(pairs) - val_size]
    hidden_size = 256
    epochs = 5

    n_hidden = 256
    axn_vocab = ["0", "1", "2" ,"!"]
    eqn_vocab = ['(', ')', '*', '+', '-', '/', '0', '1', '2', '3', '4', '5',
                 '6', '7', '8', '9']
    # print(training_input)
    # exit()
    # model = VanillaRNN(n_hidden, len(axn_vocab), len(eqn_vocab)).to(device)
    # model.load_state_dict(torch.load(os.path.normpath('models/vanilla_rnn_model_weights_256.pth')))
    # encoder1 = VanillaRNN_mod(n_hidden, len(axn_vocab), len(eqn_vocab)).to(device)
    # encoder1 = VanillaRNN_mod(n_hidden, len(axn_vocab), len(eqn _vocab)).to(device)
    # encoder1 = VanillaRNN_mod(n_hidden, len(axn_vocab), input_lang.n_words).to(device)
    encoder1 = SRNN_Softmax(hidden_size, input_lang.n_words, input_lang.n_words).to(device)
    # encoder1.W_n = model.W_y
    # encoder1.W_y = model.W_y
    # encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)

    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words,
                                   dropout_p=0.1).to(device)

    losses = []
    run_accs = []
    ind_accs = [] # List of lists
    bleus = []
    current_time = datetime.now().strftime("%H:%M:%S")
    print("Current Time =", current_time)
    val_acc = []
    for _ in range(epochs):
        loss, run_acc, ind_acc, bleu = trainIters(encoder1, attn_decoder1,
                                                  len(pairs), print_every=500)
        losses.append(loss)
        run_accs.append(run_acc)
        ind_accs.append(ind_acc)
        bleus.append(bleu)

        current_time = datetime.now().strftime("%H:%M:%S")
        print("Current Time =", current_time)

        correct = 0
        for sentence in val_pairs:
            predict, _ = evaluate(encoder1, attn_decoder1, sentence[0], input_lang, MAX_LENGTH)
            if "".join(predict[:-1]) == sentence[1]:
                correct += 1
        val_acc.append(correct/val_size)

    # print(losses, run_accs, ind_accs, bleus)

    plt.plot(val_acc)
    plt.title("Validation accuracy per epoch")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.savefig("results/val_acc.png")
    plt.close()

    plt.plot(losses)
    plt.title("Losses per epoch")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig("results/Loss.png")
    plt.close()

    plt.plot(run_accs)
    plt.title("running acc per epoch")
    plt.ylabel("Acc")
    plt.xlabel("Epoch")
    plt.savefig("results/running.png")
    plt.close()

    plt.plot(bleus)
    plt.title("bleu scores per epoch")
    plt.ylabel("Bleu")
    plt.xlabel("Epoch")
    plt.savefig("results/bleu.png")
    plt.close()

    for i in range(len(ind_accs)):
        plt.plot(ind_accs[i])
        plt.title("Accuracy per epoch")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.savefig("results/epoch_" + str(i) + ".png")
        plt.close()