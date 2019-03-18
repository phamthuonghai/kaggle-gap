#!/usr/bin/env bash

python compute_atts.py --input_file=input/gap-development.tsv --output_file=data/large-atts-gap-development.pkl
python compute_atts.py --input_file=input/gap-test.tsv --output_file=data/large-atts-gap-test.pkl
python compute_atts.py --input_file=input/gap-validation.tsv --output_file=data/large-atts-gap-validation.pkl
