'''
ELMo usage example with pre-computed and cached context independent
token representations

Below, we show usage for SQuAD where each input example consists of both
a question and a paragraph of context.
'''

import tensorflow as tf
import os
import time
import datetime
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, \
	dump_token_embeddings


class elmo():
	def __init__(self):
		self.vocab_file = 'vocab_small.txt'
		# Location of pretrained LM.  Here we use the test fixtures.
		datadir = os.path.join('pretrained')
		options_file = os.path.join(datadir, 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json')
		weight_file = os.path.join(datadir, 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5')

		# Dump the token embeddings to a file. Run this once for your dataset.
		token_embedding_file = 'elmo_token_embeddings.hdf5'
		dump_token_embeddings(
			self.vocab_file, options_file, weight_file, 
			token_embedding_file
		)
		
		self.batcher = TokenBatcher(self.vocab_file)	
		# Input placeholders to the biLM.
		self.context_token_ids = tf.placeholder('int32', shape=(None, None))
		# Build the biLM graph.
		bilm = BidirectionalLanguageModel(
			options_file,
			weight_file,
			use_character_inputs=False,
			embedding_weight_file=token_embedding_file)
		# Get ops to compute the LM embeddings.
		context_embeddings_op = bilm(self.context_token_ids)
		self.elmo_context_input = weight_layers('input', context_embeddings_op, 
			l2_coef=0.0)
		self.elmo_context_output = weight_layers('output', context_embeddings_op, 
			l2_coef=0.0)		


	def get_emb(self, tokenized_context):
		all_tokens = set(['<S>', '</S>'])
		for context_sentence in tokenized_context:
			for token in context_sentence:
				all_tokens.add(token)
		with open(self.vocab_file, 'w') as fout:
			fout.write('\n'.join(all_tokens))	
		tf.reset_default_graph()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			# Create batches of data.
			context_ids = self.batcher.batch_sentences(tokenized_context)
			# Input
			elmo_context_input_ = sess.run(
				[self.elmo_context_input['weighted_op']],
				feed_dict={self.context_token_ids: context_ids})
			# For output
			elmo_context_output_ = sess.run(
				[self.elmo_context_output['weighted_op']],
				feed_dict={self.context_token_ids: context_ids})
		return elmo_context_input_, elmo_context_output_





