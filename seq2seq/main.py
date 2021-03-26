import argparse
import os

import torch
from seq2seq.preprocess import prepare_data
from seq2seq.encoderRNN import EncoderRNN
from seq2seq.attnDecoderRNN import AttnDecoderRNN
from seq2seq.training import train_iters

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
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size, device).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, max_length, device, 0.1).to(device)

    # train_iters(encoder1, attn_decoder1, 45274, print_every=5000)
    train_iters(input_lang, output_lang, pairs, encoder1, attn_decoder1, 45274, max_length, device, teacher_forcing_ratio=0.5, print_every=5000)