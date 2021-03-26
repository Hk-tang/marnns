import argparse
import os

import torch
from preprocess import prepare_data
from encoderRNN import EncoderRNN
from attnDecoderRNN import AttnDecoderRNN
from training import train_iters
from stackRNN import SRNN_Softmax
# from evaluate import evaluate, evaluateRandomly

# run from root dir
# e.g., python3 seq2seq/main.py --data data/dataset_len100.tsv --length 100
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seq2seq")
    parser.add_argument("--data", "-d", help="Path to datafile")
    parser.add_argument("--length", "-l", help="Length of max sequence")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_lang, output_lang, pairs = prepare_data(os.path.normpath(parser.parse_args().data), "infix", 'postfix')
    max_length = round(int(parser.parse_args().length) * 1.1)
    hidden_size = 256
    # encoder1 = EncoderRNN(input_lang.n_words, hidden_size, device).to(device)
    encoder1 = SRNN_Softmax(input_lang.n_words, hidden_size, 18).to(device) # TODO: Dont hardcode vocab size
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, max_length, device, 0.1).to(device)

    # train_iters(encoder1, attn_decoder1, 45274, print_every=5000)
    train_iters(input_lang, output_lang, pairs, encoder1, attn_decoder1, 45274, max_length, device, teacher_forcing_ratio=0.5, print_every=5000)
    #evaluateRandomly(encoder1, attn_decoder1)