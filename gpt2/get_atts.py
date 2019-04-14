import os
import json
import pickle
import itertools
import sys

import fire
import numpy as np
import tensorflow as tf
import pandas as pd

import encoder
import model
sys.path.append('..')
from utils import compute_offset_no_spaces, token_length_no_space, count_chars_no_space


def encode(enc, texts):
    list_token_ids, list_tokens = [], []
    for row in texts:
        token_ids, tokens = enc.encode_with_tokens(row.lower())
        tokens = [tokens[0]] + [tok[1:] if tok[0] == '\u0120' else '##'+tok for tok in tokens[1:]]
        list_token_ids.append(token_ids)
        list_tokens.append(tokens)
    return list_token_ids, list_tokens


def get_features(data, list_tokens, att_mat):
    for _id in range(len(list_tokens)):
        # get the words A, B, Pronoun. Convert them to lower case, since we're using the uncased version of BERT
        P = data.iloc[_id]['Pronoun'].lower()
        A = data.iloc[_id]['A'].lower()
        B = data.iloc[_id]['B'].lower()

        # Ranges
        P_offset = compute_offset_no_spaces(data.iloc[_id]['Text'], data.iloc[_id]['Pronoun-offset'])
        P_length = count_chars_no_space(P)
        P_range = range(P_offset, P_offset + P_length)
        A_offset = compute_offset_no_spaces(data.iloc[_id]['Text'], data.iloc[_id]['A-offset'])
        A_length = count_chars_no_space(A)
        A_range = range(A_offset, A_offset + A_length)
        B_offset = compute_offset_no_spaces(data.iloc[_id]['Text'], data.iloc[_id]['B-offset'])
        B_length = count_chars_no_space(B)
        B_range = range(B_offset, B_offset + B_length)

        # Initialize counts
        count_chars = 0
        ids = {'A': [], 'B': [], 'P': []}
        for j, token in enumerate(list_tokens[_id]):
            # See if the character count until the current token matches the offset of any of the 3 target words
            if count_chars in P_range:
                ids['P'].append(j)
            if count_chars in A_range:
                ids['A'].append(j)
            if count_chars in B_range:
                ids['B'].append(j)
            # Update the character count
            count_chars += token_length_no_space(token)

        # Work out the label of the current piece of text
        label = 'Neither'
        if data.iloc[_id]['A-coref']:
            label = 'A'
        if data.iloc[_id]['B-coref']:
            label = 'B'

        res = {}
        for from_tok, to_tok in itertools.product(['A', 'B', 'P'], repeat=2):
            if from_tok != to_tok:
                res[from_tok + to_tok] = []
                for id_from in ids[from_tok]:
                    for id_to in ids[to_tok]:
                        res[from_tok + to_tok].append(att_mat[_id, id_from, id_to, :, :])
        res['token'] = list_tokens[_id]
        res['label'] = label
        res['ID'] = data.iloc[_id]['ID']
        yield res


def main(input_file, output_file, seed=42, model_name='117M', batch_size=1):
    np.random.seed(seed)
    tf.set_random_seed(seed)

    data = pd.read_csv(input_file, sep='\t')

    output_file = open(output_file, 'wb')
    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('../pretrained', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        output = model.model(hparams=hparams, X=context, reuse=tf.AUTO_REUSE)
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('../pretrained', model_name))
        saver.restore(sess, ckpt)

        for _id in range(0, len(data), batch_size):
            list_token_ids, list_tokens = encode(enc, data[_id:_id+batch_size]['Text'])
            out = sess.run(output, feed_dict={
                context: list_token_ids
            })
            # out['att_probs'].shape (1, 12, 12, 29, 29)
            # 'AB', 'AP', 'BA', 'BP', 'PA', 'PB', 'token', 'label', 'ID'
            # (layer, head)

            for res in get_features(data[_id:_id+batch_size], list_tokens, out['att_probs'].transpose((0, 3, 4, 1, 2))):
                pickle.dump(res, output_file)

    output_file.close()


if __name__ == '__main__':
    fire.Fire(main)
