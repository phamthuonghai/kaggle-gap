from allennlp.commands.elmo import ElmoEmbedder
from allennlp.data.tokenizers import word_tokenizer
import pandas as pd
import numpy as np
import time


def get_nearest(slot, target):
	for i in range(target, -1, -1):
		if i in slot:
			return i


def get_elmo_emb(data_name, op, wg):
	elmo = ElmoEmbedder(options_file=op, weight_file=wg, cuda_device=0)

	# data = pd.read_csv("input/gap-validation.tsv", sep = '\t')

	data = pd.read_csv(f'input/{data_name}.tsv', sep='\t')

	index = data.index
	columns = ['emb_A', 'emb_B', 'emb_P', 'label']
	emb = pd.DataFrame(index=index, columns=columns)
	emb.index.name = 'ID'

	tk = word_tokenizer.WordTokenizer()
	tokens = tk.batch_tokenize(data.Text)
	idx = []

	for i in range(len(tokens)):
		idx.append([x.idx for x in tokens[i]])
		tokens[i] = [x.text for x in tokens[i]]

	vectors = elmo.embed_sentences(tokens)

	ans = []
	for i, vector in enumerate([v for v in vectors]):
		P_l = data.iloc[i].Pronoun
		A_l = data.iloc[i].A.split()
		B_l = data.iloc[i].B.split()

		P_offset = data.iloc[i]['Pronoun-offset']
		A_offset = data.iloc[i]['A-offset']
		B_offset = data.iloc[i]['B-offset']

		if P_offset not in idx[i]:
			P_offset = get_nearest(idx[i], P_offset)
		if A_offset not in idx[i]:
			A_offset = get_nearest(idx[i], A_offset)
		if B_offset not in idx[i]:
			B_offset = get_nearest(idx[i], B_offset)

		emb_P = np.mean(vector[1:3, idx[i].index(P_offset), :], 
			axis=0, keepdims=True)

		emb_A = np.mean(vector[1:3, idx[i].index(A_offset):idx[i].index(A_offset) 
			+ len(A_l), :], axis=(1, 0), keepdims=True)
		emb_A = np.squeeze(emb_A, axis=0)

		emb_B = np.mean(vector[1:3, idx[i].index(B_offset):idx[i].index(B_offset) 
			+ len(B_l), :], axis=(1, 0), keepdims=True)
		emb_B = np.squeeze(emb_B, axis=0)

		emb_A = emb_A.reshape((1024, ))
		emb_B = emb_B.reshape((1024, ))
		emb_P = emb_P.reshape((1024, ))

		label = 'Neither'
		if data.loc[i, 'A-coref']:
			label = 'A'
		if data.loc[i, 'B-coref']:
			label = 'B'

		emb.iloc[i] = [emb_A, emb_B, emb_P, label]
	return emb

# Main
op = "pretrained/model/elmo_options.json"
wg = "pretrained/model/elmo_weights.hdf5"

print('Started at ', time.ctime())
test_emb = get_elmo_emb('gap-test', op, wg)
test_emb.to_json('data/elmo-emb-gap-test.json', 
	orient='columns')

validation_emb = get_elmo_emb('gap-validation', op, wg)
validation_emb.to_json('data/elmo-emb-gap-validation.json', 
	orient='columns')

development_emb = get_elmo_emb('gap-development', op, wg)
development_emb.to_json('data/elmo-emb-gap-development.json', 
	orient='columns')
print('Finished at ', time.ctime())