from __future__ import absolute_import, division, print_function
import itertools
import pickle as pkl
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

# OOF_NAME, DATA = 'lgbm_large_gap', 'large-atts-'
# OOF_NAME, DATA = 'lgbm_large_mgap', 'large-atts-m'

# OOF_NAME, DATA = 'lgbm_base_gap', 'base-atts-'
OOF_NAME, DATA = 'lgbm_base_mgap', 'base-atts-m'

USE_ALL = True

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def get_data(filename):
    ids = []
    data = []
    with open(filename, 'rb') as f:
        while 1:
            try:
                d = pkl.load(f)
                features = []
                ### BERT
                for lb in ['PA', 'PB', 'AP', 'BP', 'AB', 'BA']:
                    if len(d[lb]) == 0:
                        break
                    att_tensor = np.array(d[lb])
                    max_att = np.max(att_tensor, axis=0)
                    features.append(max_att[:, :].flatten())
                    
                if len(d[lb]) == 0:
                    features.append(np.zeros_like(data[-1]))
                    print("Zeros row at", d['ID'])
                ids.append(d['ID'])
                data.append(np.concatenate(features))
            except EOFError:
                break
    return ids, data

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

IDs_test, X_test = get_data('data/%stest_stage_2.pkl'%DATA)

NTEST = len(X_test)
X_test = np.array(X_test)
N_CLASSES = 3

print(OOF_NAME, DATA, X_test.shape)

KFOLD = 5
OOF_TEST = np.zeros((NTEST, N_CLASSES))
for i in range(KFOLD):
    print("FOLD #", i, end=' ')

    modelstart = time.time()
    lgb_clf = pkl.load(open("models/%s_%s.pkl"%(OOF_NAME, i), 'rb'))
    test_preds = lgb_clf.predict(X_test, raw_score=True)
    test_preds = softmax(test_preds, axis=1)
    OOF_TEST += test_preds
    print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))

OOF_TEST /= KFOLD

submission = pd.DataFrame(OOF_TEST, columns=['A', 'B', 'NEITHER'])
submission['ID'] = IDs_test
submission.set_index('ID', inplace=True)
submission.to_csv(f'output/' + OOF_NAME + '_stage2.csv')


