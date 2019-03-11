#!/usr/bin/env bash

python bert/extract_attentions.py --input_file=data/gap-development.txt --output_file=data/large-attentions-gap-development.pkl --vocab_file=pretrained/uncased_L-24_H-1024_A-16/vocab.txt --bert_config_file=pretrained/uncased_L-24_H-1024_A-16/bert_config.json --init_checkpoint=pretrained/uncased_L-24_H-1024_A-16/bert_model.ckpt --layers=-1,-2,-3,-4,-5,-6 --max_seq_length=256 --batch_size=8