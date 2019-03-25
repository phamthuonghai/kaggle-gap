#!/usr/bin/env bash

python compute_atts.py --input_file=input/mgap-development.tsv --output_file=data/large-atts-mgap-development.pkl
python compute_atts.py --input_file=input/mgap-test.tsv --output_file=data/large-atts-mgap-test.pkl
python compute_atts.py --input_file=input/mgap-validation.tsv --output_file=data/large-atts-mgap-validation.pkl
