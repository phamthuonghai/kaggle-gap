from __future__ import absolute_import, division, print_function
import itertools
import pickle as pkl
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, StratifiedKFold

# OOF_NAME, DATA = 'xgb_large_gap', 'large-atts-gap'
# OOF_NAME, DATA = 'xgb_large_mgap', 'large-atts-mgap'

# OOF_NAME, DATA = 'xgb_base_gap', 'base-atts-gap'
OOF_NAME, DATA = 'xgb_base_mgap', 'base-atts-mgap'
USE_ALL = False

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def get_data(filename):
    ids = []
    data = []
    labels = []
    with open(filename, 'rb') as f:
        while 1:
            try:
                d = pkl.load(f)
                features = []
                for lb in ['PA', 'PB', 'AP', 'BP', 'AB', 'BA']:
                    if len(d[lb]) == 0:
                        break
                    att_tensor = np.array(d[lb])
                    max_att = np.max(att_tensor, axis=0)
                    features.append(max_att[:, :].flatten())

#                     mean_att = np.mean(att_tensor, axis=0)
#                     features.append(mean_att[17:18, :].flatten())

#                     min_att = np.min(att_tensor, axis=0)
#                     features.append(min_att[17:18, :].flatten())
#                 for lb in ['AP', 'BP']:
#                     if len(d[lb]) == 0:
#                         break
#                     att_tensor = np.array(d[lb])
#                     max_att = np.max(att_tensor, axis=0)
#                     features.append(max_att[:, :].flatten())

#                     mean_att = np.mean(att_tensor, axis=0)
#                     features.append(mean_att[16:20, :].flatten())

#                     min_att = np.min(att_tensor, axis=0)
#                     features.append(min_att[16:20, :].flatten())

                if len(d[lb]) == 0:
                    features.append(np.zeros_like(data[-1]))
                    print("Zeros row at", d['ID'])
                labels.append(d['label'])
                ids.append(d['ID'])
                data.append(np.concatenate(features))
            except EOFError:
                break
    return ids, data, labels

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

IDs_train1, X_train1, Y_train1 = get_data('./data/%s-validation.pkl' %DATA)
IDs_train2, X_train2, Y_train2 = get_data('./data/%s-test.pkl'%DATA)
IDs_test, X_test, Y_test = get_data('./data/%s-development.pkl'%DATA)
if USE_ALL:
    IDs_train = IDs_train1 + IDs_train2 + IDs_test
    X_train = X_train1 + X_train2 + X_test
    Y_train = Y_train1 + Y_train2 + Y_test
else:
    IDs_train = IDs_train1 + IDs_train2
    X_train = X_train1 + X_train2
    Y_train = Y_train1 + Y_train2

NTRAIN, NTEST = len(X_train), len(X_test)
Y_train_ones = convert_label(Y_train)
Y_test_ones = convert_label(Y_test)

X, y = np.array(X_train), np.array(Y_train_ones)
X_test, y_test = np.array(X_test), np.array(Y_test_ones)
N_CLASSES = 3
EARLY_STOPPING=100
Dparam = {'objective' : "multi:softmax",
          'booster' : "gbtree",
          'eval_metric' : "mlogloss",
          'num_class': N_CLASSES,
          'nthread' : 12,
          'eta':0.05,
          'max_depth':10,
          'min_child_weight': 11,
          'gamma' :0,
          'subsample':1.0,
          'colsample_bytree':0.3,
          # 'alpha': 1.5,
          'lambda':0,
          'silent': 1}    


print(OOF_NAME, DATA, X.shape, X_test.shape)

KFOLD, SHUF, RS = 5, True, 123
OOF_TRAIN = np.zeros((NTRAIN, N_CLASSES))
OOF_TEST = np.zeros((NTEST, N_CLASSES))
val_score_list = []
kf = StratifiedKFold(n_splits=KFOLD, shuffle=SHUF, random_state=RS)
dtest = xgb.DMatrix(data=X_test)
for i, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print("FOLD #", i, end=' ')
    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[val_idx], y[val_idx]
    
    dtrain = xgb.DMatrix(X_train, y_train)
    dval = xgb.DMatrix(X_valid, y_valid)

    modelstart = time.time()
    xgb_clf = xgb.train(
        Dparam,
        dtrain,
        num_boost_round=1500,
        evals=[(dval, 'eval')],
        early_stopping_rounds=EARLY_STOPPING,
        verbose_eval=False,
    )
    pkl.dump(xgb_clf, open("models/%s_%s.pkl"%(OOF_NAME, i), 'wb'))
    test_preds = xgb_clf.predict(dtest, ntree_limit=xgb_clf.best_ntree_limit, output_margin=True)
    test_preds = softmax(test_preds, axis=1)
    err_test = log_loss(y_test, test_preds)
    OOF_TEST += test_preds
    
    val_preds = xgb_clf.predict(dval, ntree_limit=xgb_clf.best_ntree_limit, output_margin=True)
    val_preds = softmax(val_preds, axis=1)
    err = log_loss(y_valid, val_preds)
    OOF_TRAIN[val_idx] = val_preds

    print('Log Loss (val - test): %.5f - %.5f' % (err, err_test))
    val_score_list.append(err)
    # print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))

OOF_TEST /= KFOLD
# print("Loss folds", val_score_list)
print("Average %.5f - %.5f" % ((sum(val_score_list)/KFOLD), log_loss(y_test, OOF_TEST)))

submission = pd.DataFrame(OOF_TEST, columns=['A', 'B', 'NEITHER'])
submission['ID'] = IDs_test
submission.set_index('ID', inplace=True)
submission.to_csv(f'output/' + OOF_NAME + '.csv')

train_oof = pd.DataFrame(OOF_TRAIN, columns=['A', 'B', 'NEITHER'])
train_oof['ID'] = IDs_train
train_oof.set_index('ID', inplace=True)
train_oof['label'] = Y_train_ones
train_oof.to_csv(f'oof/' + OOF_NAME + '.csv')
