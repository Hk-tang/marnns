# Exploring Stack Augmented Neural Networks for Transduction — CMPUT 651 Term Project
By: Sarah Davis and Henry Tang

The objective of this project is to solve the transduction problem.
This work is a fork from the original repository where we add a Seq2Seq model and use the provided RNN implementations to replace the encoder of the Seq2Seq.

## Introduction

This repository contains all the code, instructions, and data for our term project.
The `data` directory contains the datasets used for the experiments.
The `results` directory contains the results of running our architecture on the different datasets, where each subdirectory contains the results for the different sequence lengths.

## Installation and execution

1. Activate virtual environment and install dependencies 
	```bash
    python3 -m venv venv
 
    venv\Scripts\activate.bat  # windows
    source venv/bin/activate  # unix
 
    pip install -r requirements.txt
	```
 
2. Run the program with 
    ```bash
    python seq2seq.py
	```
 
3. To test the different datasets, the source code needs to be modified
	```bash
    # seq2seq.py line 463   
 
    input_lang, output_lang, pairs = prepareData("data/dataset_len_35_40.tsv", "infix", 'postfix')
	```