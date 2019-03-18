# coding=utf-8
"""Extract pre-computed attention matrices from BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import pickle

import pandas as pd
import tensorflow as tf

from utils import compute_offset_no_spaces, count_length_no_special
from bert import modeling, tokenization
from bert.extract_utils import input_fn_builder, convert_examples_to_features, InputExample

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None, "")

flags.DEFINE_string("output_file", None, "")

flags.DEFINE_string(
    "bert_config_file", "pretrained/uncased_L-24_H-1024_A-16/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_integer(
    "max_seq_length", 256,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string(
    "init_checkpoint", "pretrained/uncased_L-24_H-1024_A-16/bert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string("vocab_file", "pretrained/uncased_L-24_H-1024_A-16/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("batch_size", 32, "Batch size for predictions.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("master", None,
                    "If using a TPU, the address of the master.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "use_one_hot_embeddings", False,
    "If True, tf.one_hot will be used for embedding lookups, otherwise "
    "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
    "since it is much faster.")


def model_fn_builder(bert_config, init_checkpoint, use_tpu, use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        input_type_ids = features["input_type_ids"]

        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=input_type_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        if mode != tf.estimator.ModeKeys.PREDICT:
            raise ValueError("Only PREDICT modes are supported: %s" % mode)

        tvars = tf.trainable_variables()
        scaffold_fn = None
        (assignment_map,
         initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
            tvars, init_checkpoint)
        if use_tpu:

            def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        # [batch_size, from_length, to_length, num_layer, num_heads]
        all_layers = model.get_all_encoder_attention_matrices()

        predictions = {
            "unique_id": unique_ids,
            'attention_matrices': all_layers
        }

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def convert_to_examples(texts):
    examples = []
    for _id, text in enumerate(texts):
        line = tokenization.convert_to_unicode(text)
        if not line:
            break
        line = line.strip()
        examples.append(InputExample(unique_id=_id, text_a=line, text_b=None))
    return examples


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        master=FLAGS.master,
        tpu_config=tf.contrib.tpu.TPUConfig(
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    data = pd.read_csv(FLAGS.input_file, sep='\t')

    examples = convert_to_examples(data['Text'])

    features = convert_examples_to_features(
        examples=examples, seq_length=FLAGS.max_seq_length, tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)

    # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        predict_batch_size=FLAGS.batch_size)

    input_fn = input_fn_builder(features=features, seq_length=FLAGS.max_seq_length)

    output_file = open(FLAGS.output_file, 'wb')

    for result in estimator.predict(input_fn, yield_single_examples=True):
        unique_id = int(result["unique_id"])
        feature = unique_id_to_feature[unique_id]

        # get the words A, B, Pronoun. Convert them to lower case, since we're using the uncased version of BERT
        P = data.loc[unique_id, 'Pronoun'].lower()
        A = data.loc[unique_id, 'A'].lower()
        B = data.loc[unique_id, 'B'].lower()

        # Ranges
        P_offset = compute_offset_no_spaces(data.loc[unique_id, 'Text'], data.loc[unique_id, 'Pronoun-offset'])
        P_length = count_length_no_special(P)
        P_range = range(P_offset, P_offset + P_length)
        A_offset = compute_offset_no_spaces(data.loc[unique_id, 'Text'], data.loc[unique_id, 'A-offset'])
        A_length = count_length_no_special(A)
        A_range = range(A_offset, A_offset + A_length)
        B_offset = compute_offset_no_spaces(data.loc[unique_id, 'Text'], data.loc[unique_id, 'B-offset'])
        B_length = count_length_no_special(B)
        B_range = range(B_offset, B_offset + B_length)

        # Initialize counts
        count_chars = 0
        ids = {'A': [], 'B': [], 'P': []}
        for j, token in enumerate(feature.tokens[1:]):
            # See if the character count until the current token matches the offset of any of the 3 target words
            if count_chars in P_range:
                ids['P'].append(j+1)
            if count_chars in A_range:
                ids['A'].append(j+1)
            if count_chars in B_range:
                ids['B'].append(j+1)
            # Update the character count
            count_chars += count_length_no_special(token)

        # Work out the label of the current piece of text
        label = 'Neither'
        if data.loc[unique_id, 'A-coref']:
            label = 'A'
        if data.loc[unique_id, 'B-coref']:
            label = 'B'

        # att_mat: [from_length, to_length, num_layer, num_heads]
        att_mat = result["attention_matrices"]

        res = {}
        for from_tok, to_tok in itertools.product(['A', 'B', 'P'], repeat=2):
            if from_tok != to_tok:
                res[from_tok + to_tok] = []
                for id_from in ids[from_tok]:
                    for id_to in ids[to_tok]:
                        res[from_tok + to_tok].append(att_mat[id_from, id_to, :, :])
        res['token'] = feature.tokens
        res['label'] = label
        res['ID'] = data.loc[unique_id, 'ID']
        pickle.dump(res, output_file)

    output_file.close()


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("init_checkpoint")
    flags.mark_flag_as_required("output_file")
    tf.app.run()
