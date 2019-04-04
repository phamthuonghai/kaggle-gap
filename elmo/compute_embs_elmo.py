from __future__ import unicode_literals, print_function
import os
import time
import numpy as np
import pandas as pd
import spacy
import nltk
from utils_elmo import compute_offset_no_spaces, count_length_no_special
from elmo_usage_token import *

nlp = spacy.blank("en")
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

elmo_embedding = elmo()

def word_tokenize(sent):
	doc = nlp(sent)
	return [token.text for token in doc]


def context_tokenize(context_sens):
	"""
	Prepare data for run ELMo
	Input: list of sentences
	Output: 
	- raw_context: list of sentences with tokenize
	"there." = "there ."
	- tokenized_context: list of sentences, 
	each sentence is a list of tokens
	"""
	tokenized_context = []
	raw_context = []
	for p in context_sens:
		tokens = word_tokenize(p)
		tokenized_context.append(tokens)
		sen = ' '.join(tokens)
		raw_context.append(sen)
	return raw_context, tokenized_context


def find_position(context_sens, tokenized_context, word, word_offset):
	"""
	Find the index to get embedding after run ELMo
	Output: idx_1, list_idx_2
	- idx_1: index for sentence
	- list_index_2: index for token in sentence
	because word have more than one token then should use list
	"""
	list_ = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	list_idx_2 = []
	start = 0
	end = 0
	for sen in context_sens:
		start = end
		end += count_length_no_special(sen)
		if word_offset >= start and word_offset < end:
			idx_1 = context_sens.index(sen)
			words = word.split()
			count = start
			sen_in_para_token = tokenized_context[idx_1]
			for token in sen_in_para_token:
				if count == word_offset:
					index = sen_in_para_token.index(token)
					list_idx_2 = list_[0:len(words)]
					list_idx_2 = [ele+index for ele in list_idx_2]
					break
				count += count_length_no_special(token)
	return idx_1, list_idx_2


def run_elmo(data_name, filename):
	fout = open(filename, 'w')
	# data_name = "gap-validation"
	data = pd.read_csv(f'input/{data_name}.tsv', sep='\t')
	text = data['Text']
	text.to_csv(f'data/{data_name}.txt', index=False, header=False)

	index = data.index
	columns = ['emb_A', 'emb_B', 'emb_P', 'label']
	emb = pd.DataFrame(index=index, columns=columns)
	emb.index.name = 'ID'
	
	for i in range(len(data)):
		A = data.loc[i, 'A']
		B = data.loc[i, 'B']
		P = data.loc[i, 'Pronoun']
				# Get offset
		P_offset = compute_offset_no_spaces(data.loc[i, 'Text'], 
			data.loc[i, 'Pronoun-offset'])
		A_offset = compute_offset_no_spaces(data.loc[i, 'Text'], 
			data.loc[i, 'A-offset'])
		B_offset = compute_offset_no_spaces(data.loc[i, 'Text'], 
			data.loc[i, 'B-offset'])
		# process text
		context_text = data.loc[i, 'Text']
		context_sens = sentence_tokenizer.tokenize(context_text)

		raw_context, tokenized_context = context_tokenize(context_sens)
		elmo_context_input_, elmo_context_output_ = \
			elmo_embedding.get_emb(tokenized_context)

		A_idx_1, A_list_idx_2 = find_position(context_sens, 
			tokenized_context, A, A_offset)
		B_idx_1, B_list_idx_2 = find_position(context_sens, 
			tokenized_context, B, B_offset)
		P_idx_1, P_list_idx_2 = find_position(context_sens, 
			tokenized_context, P, P_offset)
		   
		emb_A = np.zeros(1024)
		emb_B = np.zeros(1024)
		emb_P = np.zeros(1024)
		emb_all = elmo_context_input_[0]
		dimen_2 = emb_all.shape[1]

		emb_P = emb_all[P_idx_1][P_list_idx_2[0]][:]

		for idx in A_list_idx_2:
			if idx >= dimen_2:
				print("Something wrong with sentence segmentation!!!")
				print(idx)
				print(dimen_2)
				fout.write('\n'.join(context_sens))	
				fout.write("\n")
				A_idx_1 += 1
				idx -= dimen_2
			emb_A += emb_all[A_idx_1][idx][:]
		emb_A /= len(A_list_idx_2)

		for idx in B_list_idx_2:
			if idx >= dimen_2:
				print("Something wrong with sentence segmentation!!!")
				print(idx)
				print(dimen_2)
				fout.write('\n'.join(context_sens))
				B_idx_1 += 1
				idx -= dimen_2			
			emb_B += emb_all[B_idx_1][idx][:]
		emb_B /= len(B_list_idx_2)
		# Work out the label of the current piece of text
		label = 'Neither'
		if data.loc[i, 'A-coref']:
			label = 'A'
		if data.loc[i, 'B-coref']:
			label = 'B'

		emb.iloc[i] = [emb_A, emb_B, emb_P, label]
	fout.close()
	return emb


print('Started at ', time.ctime())
test_emb = run_elmo('gap-test', 'data/gap-test-wrong')
test_emb.to_json('data/elmo-emb-gap-test.json', 
	orient='columns')

validation_emb = run_elmo('gap-validation', 
	'data/gap-validation-wrong')
validation_emb.to_json('data/elmo-emb-gap-validation.json', 
	orient='columns')

development_emb = run_elmo('gap-development', 
	'data/gap-development-wrong')
development_emb.to_json('data/elmo-emb-gap-development.json', 
	orient='columns')
print('Finished at ', time.ctime())
