#!/usr/bin/env bash

export DATA=test_stage_2
export MODEL="uncased_L-24_H-1024_A-16"
export MODEL_NAME=large
export PARAMS="--bert_config_file=pretrained/${MODEL}/bert_config.json --init_checkpoint=pretrained/${MODEL}/bert_model.ckpt --vocab_file=pretrained/${MODEL}/vocab.txt"
python compute_atts.py --input_file=input/${DATA}.tsv --output_file=data/${MODEL_NAME}-atts-${DATA}.pkl ${PARAMS}

export DATA=mtest_stage_2
export PARAMS="--bert_config_file=pretrained/${MODEL}/bert_config.json --init_checkpoint=pretrained/${MODEL}/bert_model.ckpt --vocab_file=pretrained/${MODEL}/vocab.txt"
python compute_atts.py --input_file=input/${DATA}.tsv --output_file=data/${MODEL_NAME}-atts-${DATA}.pkl ${PARAMS}

export MODEL="uncased_L-12_H-768_A-12"
export MODEL_NAME=base
export PARAMS="--bert_config_file=pretrained/${MODEL}/bert_config.json --init_checkpoint=pretrained/${MODEL}/bert_model.ckpt --vocab_file=pretrained/${MODEL}/vocab.txt"
python compute_atts.py --input_file=input/${DATA}.tsv --output_file=data/${MODEL_NAME}-atts-${DATA}.pkl ${PARAMS}

export DATA=test_stage_2
export PARAMS="--bert_config_file=pretrained/${MODEL}/bert_config.json --init_checkpoint=pretrained/${MODEL}/bert_model.ckpt --vocab_file=pretrained/${MODEL}/vocab.txt"
python compute_atts.py --input_file=input/${DATA}.tsv --output_file=data/${MODEL_NAME}-atts-${DATA}.pkl ${PARAMS}
