import argparse

import pandas as pd
import nltk


def trim_paragraph(row):
    text = row['Text']
    sents = nltk.sent_tokenize(text)
    min_offset = min(row['Pronoun-offset'], row['A-offset'], row['B-offset'])
    max_offset = max(row['Pronoun-offset'], row['A-offset'], row['B-offset'])

    sum_chars = 0
    start_id = None
    for sent in sents:
        sent_len = len(sent)

        # Align
        while text[sum_chars:sum_chars + sent_len] != sent:
            sum_chars += 1

        if min_offset < sum_chars + sent_len and start_id is None:
            start_id = sum_chars
        sum_chars += sent_len
        if max_offset < sum_chars:
            break
    return start_id, sum_chars


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    args = parser.parse_args()

    data = pd.read_csv(args.input_file, sep='\t', index_col='ID')
    data['Cut'] = data.apply(trim_paragraph, axis=1)
    data['Text'] = data.apply(lambda row: row['Text'][row['Cut'][0]:row['Cut'][1]], axis=1)
    data['Pronoun-offset'] = data.apply(lambda row: row['Pronoun-offset']-row['Cut'][0], axis=1)
    data['A-offset'] = data.apply(lambda row: row['A-offset']-row['Cut'][0], axis=1)
    data['B-offset'] = data.apply(lambda row: row['B-offset']-row['Cut'][0], axis=1)
    data.to_csv(args.output_file, sep='\t')
