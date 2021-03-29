import time
import random
import math
import torch
import torch.nn as nn
from torch import optim


SOS_token = 0
EOS_token = 1


def train_iters(input_lang, output_lang, pairs, encoder, decoder, n_iters, max_length, device, teacher_forcing_ratio, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    print_total_accuracy = 0
    num_correct = 0
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensors_from_pair(input_lang, output_lang, random.choice(pairs), device) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    j = 100

    for i in range(1, n_iters + 1):
        training_pair = training_pairs[i - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss, acc, results = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length, device, teacher_forcing_ratio, output_lang)
        print_loss_total += loss
        plot_loss_total += loss
        num_correct += acc
        

        if i % print_every == 0 or i == n_iters:
            print_loss_avg = print_loss_total / print_every
            print_total_accuracy = num_correct/print_every
            print_loss_total = 0
            num_correct = 0
            print('%s (%d %d%%) loss: %.4f \naverage acc per sample %.4f' % (time_since(start, i / n_iters),
                                         i, i / n_iters * 100,
                                         print_loss_avg, print_total_accuracy))
            print("Sample:\nActual:", results[0], '\nPredicted:', results[1])

        if i % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            print_total_accuracy = 0


def tensors_from_pair(input_lang, output_lang, pair, device):
    input_tensor = tensor_from_sentence(input_lang, pair[0], device)
    target_tensor = tensor_from_sentence(output_lang, pair[1], device)
    return input_tensor, target_tensor


def tensor_from_sentence(lang, sentence, device):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length, device, teacher_forcing_ratio, output_lang):
    encoder_hidden = encoder.initHidden()
    encoder_stack = encoder.initStack()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size,
                                  device=device)

    loss = 0

    for ei in range(input_length):
        # encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        # print("input_tensor[ei]", input_tensor[ei])
        encoder_output, encoder_hidden, encoder_stack = encoder(input_tensor[ei], encoder_hidden, encoder_stack) #line changed
        encoder_outputs[ei] = encoder_output[0, 0]
    
    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = False #if random.random() < teacher_forcing_ratio else False
    answer = ''
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            answer += output_lang.index2word[topi.item()]
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
        

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            answer += output_lang.index2word[topi.item()]
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            # if decoder_input.item() == EOS_token:
            #     break

    a = [output_lang.index2word[i.item()] for i in target_tensor]
    a = ''.join(a)
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    a = a.strip()
    answer = answer.strip()
    acc = 0
    for i in range(len(a)):
        if a[i] == answer[i]:
            acc += 1
    l = loss.item() / target_length
    acc/=len(a)
    results = [a, answer]
    # acc = 1 if a.strip() == answer.strip() else 0
    return l, acc, results


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
