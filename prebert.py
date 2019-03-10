import time

import numpy as np
import pandas as pd
from keras import callbacks as kc
from keras import models, layers, regularizers, optimizers
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

dense_layer_sizes = [37]
dropout_rate = 0.6
learning_rate = 0.001
n_fold = 5
batch_size = 32
epochs = 1000
patience = 100
# n_test = 100
lambd = 0.1  # L2 regularization


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
    model = models.Model(input=X_input, output=X, name='classif_model')
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
    X = np.zeros((len(embeddings), 3 * 768))
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

    return X, Y


# Read development embeddings from json file - this is the output of Bert
development = pd.read_json('data/emb-gap-development.json')
X_development, Y_development = parse_json(development)

validation = pd.read_json('data/emb-gap-validation.json')
X_validation, Y_validation = parse_json(validation)

test = pd.read_json('data/emb-gap-test.json')
X_test, Y_test = parse_json(test)

# There may be a few NaN values, where the offset of a target word is greater than the max_seq_length of BERT.
# They are very few, so I'm just dropping the rows.
remove_test = [row for row in range(len(X_test)) if np.sum(np.isnan(X_test[row]))]
X_test = np.delete(X_test, remove_test, 0)
Y_test = np.delete(Y_test, remove_test, 0)

remove_validation = [row for row in range(len(X_validation)) if np.sum(np.isnan(X_validation[row]))]
X_validation = np.delete(X_validation, remove_validation, 0)
Y_validation = np.delete(Y_validation, remove_validation, 0)

# We want predictions for all development rows. So instead of removing rows, make them 0
remove_development = [row for row in range(len(X_development)) if np.sum(np.isnan(X_development[row]))]
X_development[remove_development] = np.zeros(3 * 768)

# Will train on data from the gap-test and gap-validation files, in total 2454 rows
X_train = np.concatenate((X_test, X_validation), axis=0)
Y_train = np.concatenate((Y_test, Y_validation), axis=0)

# Will predict probabilities for data from the gap-development file; initializing the predictions
prediction = np.zeros((len(X_development), 3))  # testing predictions

# Training and cross-validation
folds = KFold(n_splits=n_fold, shuffle=True, random_state=3)
scores = []
for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train)):
    # split training and validation data
    print('Fold', fold_n, 'started at', time.ctime())
    X_tr, X_val = X_train[train_index], X_train[valid_index]
    Y_tr, Y_val = Y_train[train_index], Y_train[valid_index]

    # Define the model, re-initializing for each fold
    classif_model = build_mlp_model([X_train.shape[1]])
    classif_model.compile(optimizer=optimizers.Adam(lr=learning_rate), loss='categorical_crossentropy')
    callbacks = [kc.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)]

    # train the model
    classif_model.fit(x=X_tr, y=Y_tr, epochs=epochs, batch_size=batch_size, callbacks=callbacks,
                      validation_data=(X_val, Y_val), verbose=0)

    # make predictions on validation and test data
    pred_valid = classif_model.predict(x=X_val, verbose=0)
    pred = classif_model.predict(x=X_development, verbose=0)

    # oof[valid_index] = pred_valid.reshape(-1,)
    scores.append(log_loss(Y_val, pred_valid))
    prediction += pred
prediction /= n_fold

# Print CV scores, as well as score on the test data
print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
print(scores)
print('Test score:', log_loss(Y_development, prediction))

# Write the prediction to file for submission
submission = pd.read_csv('input/sample_submission_stage_1.csv', index_col='ID')
submission['A'] = prediction[:, 0]
submission['B'] = prediction[:, 1]
submission['NEITHER'] = prediction[:, 2]
submission.to_csv('output/submission_bert.csv')
