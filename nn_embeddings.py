import pickle as pkl
import time

import numpy as np
import pandas as pd
from keras import callbacks
from keras import regularizers, optimizers
from keras.layers import *
from keras.models import *
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

OOF_NAME, DATA, DIM = 'nn_large_emb_gap', 'large-emb-gap', 1024
# OOF_NAME, DATA, DIM = 'nn_base_emb_gap', 'base-emb-gap', 768
# OOF_NAME, DATA, DIM = 'elmo_emb_gap', 'elmo-emb-gap', 1024
USE_ALL = False

model_name = 'pretrained-mlp'
learning_rate = 0.001
batch_size = 128
epochs = 1000
patience = 100
# n_test = 100
lambd = 0.1  # L2 regularization


class End2End_NCR():
    
    def __init__(self, word_input_shape): 
        
        self.word_input_shape = word_input_shape
        self.hidden_dim   = 150
        
    def build(self):
        
        A, B, P = Input((self.word_input_shape,)), Input((self.word_input_shape,)), Input((self.word_input_shape,))
        inputs = [A, B, P]

        
        self.ffnn = Sequential([Dense(self.hidden_dim, use_bias=True),
                                     Activation('relu'),
                                     Dropout(rate=0.2, seed = 7),
                                     Dense(1, activation='linear')])

        PA = Multiply()([inputs[0], inputs[2]])
        PB = Multiply()([inputs[1], inputs[2]])

        PA = Concatenate(axis=-1)([P, A, PA])
        PB = Concatenate(axis=-1)([P, B, PB])
        PA_score = self.ffnn(PA)
        PB_score = self.ffnn(PB)
        # Fix the Neither to score 0.
        score_e  = Lambda(lambda x: K.zeros_like(x))(PB_score)
        
        #Final Output
        output = Concatenate(axis=-1)([PA_score, PB_score, score_e])
        output = Activation('softmax')(output)        
        model = Model(inputs, output)
        
        return model
    
def parse_json(embeddings):
    """
    Parses the embeddings given by BERT, and suitably formats them to be passed to the MLP model

    Input: embeddings (DataFrame) containing contextual embeddings from BERT, also labels for the classification problem
    columns: "emb_A": contextual embedding for the word A
             "emb_B": contextual embedding for the word B
             "emb_P": contextual embedding for the pronoun
             "label": the answer to the coreference problem: "A", "B" or "NEITHER"

    Output: X (numpy array) for each line in the GAP file, the concatenation of the embeddings of the target words
            Y (numpy array) for each line in the GAP file, the one-hot encoded answer to the coreference problem
    """
    embeddings.sort_index(
        inplace=True)  # Sorting the DataFrame, because reading from the json file messed with the order
    N = len(embeddings)
    Y = np.zeros((N, 3))
    all_P = np.zeros((N, DIM))
    all_A = np.zeros((N, DIM))
    all_B = np.zeros((N, DIM))
    # Concatenate features
    for i in range(len(embeddings)):
        all_A[i] = np.array(embeddings.loc[i, 'emb_A'])
        all_B[i] = np.array(embeddings.loc[i, 'emb_B'])
        all_P[i] = np.array(embeddings.loc[i, 'emb_P'])

    # One-hot encoding for labels
    for i in range(len(embeddings)):
        label = embeddings.loc[i, 'label']
        if label == 'A':
            Y[i, 0] = 1
        elif label == 'B':
            Y[i, 1] = 1
        else:
            Y[i, 2] = 1
    
    np.nan_to_num(all_P, copy=False)
    np.nan_to_num(all_A, copy=False)
    np.nan_to_num(all_B, copy=False)
    return [all_A, all_B, all_P], Y

def get_data(filename):
    ids = []
    labels = []
    with open(filename, 'rb') as f:
        while 1:
            try:
                d = pkl.load(f)
                labels.append(d['label'])
                ids.append(d['ID'])
            except EOFError:
                break
    return ids, labels

def convert_label(Y_train):
    Y_train_ones = []
    for y in Y_train:
        if y == 'A':
            Y_train_ones.append(0)
        elif y == 'B':
            Y_train_ones.append(1)
        else:
            Y_train_ones.append(2)
    return Y_train_ones

IDs_train1, Y_train1 = get_data('data/base-atts-gap-validation.pkl')
IDs_train2, Y_train2  = get_data('data/base-atts-gap-test.pkl')
IDs_test, Y_test  = get_data('data/base-atts-gap-development.pkl')

if USE_ALL:
    IDs_train = IDs_train1 + IDs_train2 + IDs_test
    Y_train = Y_train1 + Y_train2 + Y_test
else:
    IDs_train = IDs_train1 + IDs_train2
    Y_train = Y_train1 + Y_train2

Y_train_ones = convert_label(Y_train)

# Read development embeddings from json file - this is the output of Bert
development = pd.read_json('data/%s-development.json' % DATA)
validation = pd.read_json('data/%s-validation.json'% DATA)
test = pd.read_json('data/%s-test.json'% DATA)

new_train = pd.concat([validation, test])
new_train = new_train.reset_index(drop=True)

X_train, Y_train = parse_json(new_train)
X_test, Y_test = parse_json(development)


# Will predict probabilities for data from the gap-development file; initializing the predictions
NTRAIN, NTEST = len(X_train[0]), len(X_test[0])
print(OOF_NAME, DATA, X_train[0].shape, X_test[0].shape)
N_CLASSES = 3
OOF_TRAIN = np.zeros((NTRAIN, N_CLASSES))
OOF_TEST = np.zeros((NTEST, N_CLASSES))
val_score_list = []

# End2End_NCR(X_train[0].shape[1]).build().summary()
# Training and cross-validation
KFOLD, SHUF, RS = 5, True, 123
kf = StratifiedKFold(n_splits=KFOLD, shuffle=SHUF, random_state=RS)
for fold_n, (train_index, valid_index) in enumerate(kf.split(X_train[0], Y_train_ones)):
    # split training and validation data
    print('Fold #%d' % fold_n, end=' ')
    X_tr = [x[train_index] for x in X_train]
    X_val = [x[valid_index] for x in X_train]
    Y_tr, Y_val = Y_train[train_index], Y_train[valid_index]

    # Define the model, re-initializing for each fold
    classif_model = End2End_NCR(X_train[0].shape[1]).build()
    classif_model.compile(optimizer=optimizers.Adam(lr=learning_rate), loss='categorical_crossentropy')
    file_path = "models/%s_%s.hdf5" % (OOF_NAME, fold_n)
    check_point = callbacks.ModelCheckpoint(file_path, monitor = "val_loss", verbose = 0, save_best_only = True, mode = "min")
    early_stop = callbacks.EarlyStopping(monitor = "val_loss", mode = "min", patience=100)

    # train the model
    classif_model.fit(x=X_tr, y=Y_tr, epochs=epochs, batch_size=batch_size, callbacks=[check_point, early_stop],
                      validation_data=(X_val, Y_val), verbose=0)

    # make predictions on validation and test data
    pred_valid = classif_model.predict(x=X_val, verbose=0)
    test_preds = classif_model.predict(x=X_test, verbose=0)

    err_test = log_loss(Y_test, test_preds)
    err = log_loss(Y_val, pred_valid)
    # OOF_TRAIN[valid_index] = pred_valid.reshape(-1,)
    print('Log Loss (val - test): %.5f - %.5f' % (err, err_test))
    val_score_list.append(err)
    OOF_TEST += test_preds
    
OOF_TEST /= KFOLD

# Print CV scores, as well as score on the test data
print("Average %.5f - %.5f" % (np.mean(val_score_list), log_loss(Y_test, OOF_TEST)))

# Write the prediction to file for submission
submission = pd.read_csv('input/sample_submission_stage_1.csv', index_col='ID')
submission['A'] = OOF_TEST[:, 0]
submission['B'] = OOF_TEST[:, 1]
submission['NEITHER'] = OOF_TEST[:, 2]
submission.to_csv(f'output/{model_name}.csv')
