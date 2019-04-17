#!/usr/bin/env bash

export DATA=gap
#export MODEL="uncased_L-12_H-768_A-12"
export MODEL="uncased_L-24_H-1024_A-16"
export MODEL_NAME=large

export PARAMS="--bert_config_file=pretrained/${MODEL}/bert_config.json --init_checkpoint=pretrained/${MODEL}/bert_model.ckpt --vocab_file=pretrained/${MODEL}/vocab.txt"
python compute_atts_onames.py --input_file=input/${DATA}-development.tsv --output_file=data/${MODEL_NAME}-onames-atts-${DATA}-development.pkl ${PARAMS}
python compute_atts_onames.py --input_file=input/${DATA}-test.tsv --output_file=data/${MODEL_NAME}-onames-atts-${DATA}-test.pkl ${PARAMS}
python compute_atts_onames.py --input_file=input/${DATA}-validation.tsv --output_file=data/${MODEL_NAME}-onames-atts-${DATA}-validation.pkl ${PARAMS}
