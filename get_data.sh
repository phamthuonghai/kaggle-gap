#!/usr/bin/env bash

# Pretrained BERT models
mkdir pretrained
cd pretrained
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip
unzip uncased_L-24_H-1024_A-16.zip
cd ..

# Input data
mkdir input
cd input
wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv
wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-validation.tsv
wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv
kaggle competitions download -c gendered-pronoun-resolution
unzip test_stage_1.tsv.zip
ls
cd ..

mkdir data
mkdir output
