#!/usr/bin/env bash

# Pretrained ELMo models
# mkdir pretrained
# cd pretrained
# wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5
# wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json
# cd ..

# # Input data
# mkdir input
# cd input
# wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv
# wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-validation.tsv
# wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv
# kaggle competitions download -c gendered-pronoun-resolution
# unzip test_stage_1.tsv.zip
# ls
# cd ..

# mkdir data
python compute_embs_elmo.py
