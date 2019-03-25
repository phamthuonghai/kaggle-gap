import os
import time

import numpy as np
import pandas as pd

from utils import compute_offset_no_spaces, token_length_no_space, count_chars_no_space


def run_bert(data_name):
    """
    Runs a forward propagation of BERT on input text, extracting contextual word embeddings
    Input: data, a pandas DataFrame containing the information in one of the GAP files

    Output: emb, a pandas DataFrame containing contextual embeddings for the words A, B and Pronoun.
            Each embedding is a numpy array of shape (1024)
    columns: "emb_A": the embedding for word A
             "emb_B": the embedding for word B
             "emb_P": the embedding for the pronoun
             "label": the answer to the coreference problem: "A", "B" or "NEITHER"
    """
    model_name = 'uncased_L-24_H-1024_A-16'
    data = pd.read_csv(f'input/{data_name}.tsv', sep='\t')
    text = data['Text']
    text.to_csv(f'data/{data_name}.txt', index=False, header=False)

    os.system(f'python bert/extract_features.py \
    --input_file=data/{data_name}.txt \
    --output_file=data/large-all-emb-{data_name}.jsonl \
    --vocab_file=pretrained/{model_name}/vocab.txt \
    --bert_config_file=pretrained/{model_name}/bert_config.json \
    --init_checkpoint=pretrained/{model_name}/bert_model.ckpt \
    --layers=-1 \
    --max_seq_length=256 \
    --batch_size=8')

    bert_output = pd.read_json(f'data/large-all-emb-{data_name}.jsonl', lines=True)

    index = data.index
    columns = ['emb_A', 'emb_B', 'emb_P', 'label']
    emb = pd.DataFrame(index=index, columns=columns)
    emb.index.name = 'ID'

    for i in range(len(data)):  # For each line in the data file
        # get the words A, B, Pronoun. Convert them to lower case, since we're using the uncased version of BERT
        # P = data.loc[i, 'Pronoun'].lower()
        A = data.loc[i, 'A'].lower()
        B = data.loc[i, 'B'].lower()

        # For each word, find the offset not counting spaces. This is necessary for comparison with the output of BERT
        P_offset = compute_offset_no_spaces(data.loc[i, 'Text'], data.loc[i, 'Pronoun-offset'])
        A_offset = compute_offset_no_spaces(data.loc[i, 'Text'], data.loc[i, 'A-offset'])
        B_offset = compute_offset_no_spaces(data.loc[i, 'Text'], data.loc[i, 'B-offset'])
        # Figure out the length of A, B, not counting spaces or special characters
        A_length = count_chars_no_space(A)
        B_length = count_chars_no_space(B)

        # Initialize embeddings with zeros
        emb_A = np.zeros(1024)
        emb_B = np.zeros(1024)
        emb_P = np.zeros(1024)

        # Initialize counts
        count_chars = 0
        cnt_A, cnt_B, cnt_P = 0, 0, 0

        features = pd.DataFrame(
            bert_output.loc[i, 'features'])  # Get the BERT embeddings for the current line in the data file
        # Iterate over the BERT tokens for the current line; we skip over the first 2 token
        # which don't correspond to words
        for j in range(2, len(features)):
            token = features.loc[j, 'token']

            # See if the character count until the current token matches the offset of any of the 3 target words
            if count_chars == P_offset:
                # print(token)
                emb_P += np.array(features.loc[j, 'layers'][0]['values'])
                cnt_P += 1
            if count_chars in range(A_offset, A_offset + A_length):
                # print(token)
                emb_A += np.array(features.loc[j, 'layers'][0]['values'])
                cnt_A += 1
            if count_chars in range(B_offset, B_offset + B_length):
                # print(token)
                emb_B += np.array(features.loc[j, 'layers'][0]['values'])
                cnt_B += 1
            # Update the character count
            count_chars += token_length_no_space(token)
        # Taking the average between tokens in the span of A or B, so divide the current value by the count
        emb_A /= cnt_A
        emb_B /= cnt_B

        # Work out the label of the current piece of text
        label = 'Neither'
        if data.loc[i, 'A-coref']:
            label = 'A'
        if data.loc[i, 'B-coref']:
            label = 'B'

        # Put everything together in emb
        emb.iloc[i] = [emb_A, emb_B, emb_P, label]

    return emb


print('Started at ', time.ctime())
test_emb = run_bert('gap-test')
test_emb.to_json('data/large-emb-gap-test.json', orient='columns')

validation_emb = run_bert('gap-validation')
validation_emb.to_json('data/large-emb-gap-validation.json', orient='columns')

development_emb = run_bert('gap-development')
development_emb.to_json('data/large-emb-gap-development.json', orient='columns')
print('Finished at ', time.ctime())
