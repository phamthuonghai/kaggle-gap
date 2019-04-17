import time
import pickle as pkl

import numpy as np
import pandas as pd
from keras import callbacks as kc
from keras import models, layers, regularizers, optimizers
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

# OOF_NAME, DATA, DIM = 'nn_large_emb_gap', 'large-emb-gap', 1024
OOF_NAME, DATA, DIM = 'nn_base_emb_gap', 'base-emb-gap', 768
# OOF_NAME, DATA, DIM = 'elmo_emb_gap', 'elmo-emb-gap', 1024
USE_ALL = False

dense_layer_sizes = [37]
dropout_rate = 0.6
learning_rate = 0.001
n_fold = 5
batch_size = 32
epochs = 1000
patience = 100
# n_test = 100
lambd = 0.1  # L2 regularization

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

def build_mlp_model(input_shape):
    X_input = layers.Input(input_shape)

    # First dense layer
    X = layers.Dense(dense_layer_sizes[0], name='dense0')(X_input)
    X = layers.BatchNormalization(name='bn0')(X)
    X = layers.Activation('relu')(X)
    X = layers.Dropout(dropout_rate, seed=7)(X)

    # Second dense layer
    # 	X = layers.Dense(dense_layer_sizes[0], name = 'dense1')(X)
    # 	X = layers.BatchNormalization(name = 'bn1')(X)
    # 	X = layers.Activation('relu')(X)
    # 	X = layers.Dropout(dropout_rate, seed = 9)(X)

    # Output layer
    X = layers.Dense(3, name='output', kernel_regularizer=regularizers.l2(lambd))(X)
    X = layers.Activation('softmax')(X)

    # Create model
    model = models.Model(inputs=X_input, outputs=X, name='classif_model')
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
    X = np.zeros((len(embeddings), 3 * DIM))
    Y = np.zeros((len(embeddings), 3))

    # Concatenate features
    for i in range(len(embeddings)):
        A = np.array(embeddings.loc[i, 'emb_A'])
        B = np.array(embeddings.loc[i, 'emb_B'])
        P = np.array(embeddings.loc[i, 'emb_P'])
        X[i] = np.concatenate((A, B, P))

    # One-hot encoding for labels
    for i in range(len(embeddings)):
        label = embeddings.loc[i, 'label']
        if label == 'A':
            Y[i, 0] = 1
        elif label == 'B':
            Y[i, 1] = 1
        else:
            Y[i, 2] = 1
    np.nan_to_num(X, copy=False)
    return X, Y


IDs_train1, Y_train1 = get_data('data/large-atts-gap-validation.pkl')
IDs_train2, Y_train2  = get_data('data/large-atts-gap-test.pkl')
IDs_test, Y_test  = get_data('data/large-atts-gap-development.pkl')

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

NTRAIN, NTEST = len(X_train), len(X_test)
print(OOF_NAME, DATA, X_train.shape, X_test.shape)
N_CLASSES = 3
OOF_TRAIN = np.zeros((NTRAIN, N_CLASSES))
OOF_TEST = np.zeros((NTEST, N_CLASSES))
val_score_list = []

# Training and cross-validation
KFOLD, SHUF, RS = 5, True, 123
kf = StratifiedKFold(n_splits=KFOLD, shuffle=SHUF, random_state=RS)
for fold_n, (train_index, valid_index) in enumerate(kf.split(X_train, Y_train_ones)):
    # split training and validation data
    print('Fold #%d' % fold_n, end=' ')
    X_tr, X_val = X_train[train_index], X_train[valid_index]
    Y_tr, Y_val = Y_train[train_index], Y_train[valid_index]

    # Define the model, re-initializing for each fold
    classif_model = build_mlp_model([X_train.shape[1]])
    classif_model.compile(optimizer=optimizers.Adam(lr=learning_rate), loss='categorical_crossentropy')

    file_path = "models/%s_%s.hdf5" % (OOF_NAME, fold_n)
    check_point = kc.ModelCheckpoint(file_path, monitor = "val_loss", verbose = 0, save_best_only = True, mode = "min")
    early_stop = kc.EarlyStopping(monitor = "val_loss", mode = "min", patience=100, restore_best_weights = True)

    # train the model
    classif_model.fit(x=X_tr, y=Y_tr, epochs=epochs, batch_size=batch_size, callbacks=[check_point, early_stop],
                      validation_data=(X_val, Y_val), verbose=0)

    # make predictions on validation and test data
    pred_valid = classif_model.predict(x=X_val, verbose=0)
    test_preds = classif_model.predict(x=X_test, verbose=0)

    err_test = log_loss(Y_test, test_preds)
    err = log_loss(Y_val, pred_valid)
#     print(pred_valid.shape)
#     OOF_TRAIN[valid_index] = pred_valid
    
    print('Log Loss (val - test): %.5f - %.5f' % (err, err_test))
    val_score_list.append(err)
    OOF_TEST += test_preds
    
OOF_TEST /= KFOLD


print("Average %.5f - %.5f" % (np.mean(val_score_list), log_loss(Y_test, OOF_TEST)))

submission = pd.DataFrame(OOF_TEST, columns=['A', 'B', 'NEITHER'])
submission['ID'] = IDs_test
submission.set_index('ID', inplace=True)
submission.to_csv(f'output/' + OOF_NAME + '.csv')

train_oof = pd.DataFrame(OOF_TRAIN, columns=['A', 'B', 'NEITHER'])
train_oof['ID'] = IDs_train
train_oof.set_index('ID', inplace=True)
train_oof['label'] = Y_train_ones
train_oof.to_csv(f'oof/' + OOF_NAME + '.csv')

