from __future__ import absolute_import, division, print_function
import itertools
import pickle as pkl
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, StratifiedKFold

# OOF_NAME, DATA = 'lgbm_large_gap', 'large-atts-gap'
# OOF_NAME, DATA = 'lgbm_large_mgap', 'large-atts-mgap'

# OOF_NAME, DATA = 'lgbm_base_gap', 'base-atts-gap'
# OOF_NAME, DATA = 'lgbm_base_mgap', 'base-atts-mgap'

# OOF_NAME, DATA = 'lgbm_gpt2_gap', 'gpt2-atts-gap'


# OOF_NAME, DATA = 'lgbm_clarge_gap', 'clarge-atts-gap'
# OOF_NAME, DATA = 'lgbm_mbase_gap', 'mbase-atts-gap'
# OOF_NAME, DATA = 'lgbm_mcbase_gap', 'mcbase-atts-gap'

USE_ALL = True

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
                ### BERT
                for lb in ['PA', 'PB', 'AP', 'BP', 'AB', 'BA']:
                    if len(d[lb]) == 0:
                        break
                    att_tensor = np.array(d[lb])
                    max_att = np.max(att_tensor, axis=0)
                    features.append(max_att[:, :].flatten())

                ### GPT2
#                 for lb, lb1 in [('PA', 'AP'), ('PB','BP'), ('BA', 'AB')]:
#                     if len(d[lb]) == 0 and len(d[lb1]) == 0:
#                         break
#                     att_tensor = np.array(d[lb])
#                     max_att = np.max(att_tensor, axis=0)
#                     att_tensor1 = np.array(d[lb1])
#                     max_att1 = np.max(att_tensor1, axis=0)
#                     new_max_att = max_att - max_att1
#                     if new_max_att.sum() == 0:
#                         print("Zeros", lb, len(data))
#                     features.append(new_max_att[:, :].flatten())

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

IDs_train1, X_train1, Y_train1 = get_data('data/%s-validation.pkl' %DATA)
IDs_train2, X_train2, Y_train2 = get_data('data/%s-test.pkl'%DATA)
IDs_test, X_test, Y_test = get_data('data/%s-development.pkl'%DATA)
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
pkl.dump(y_test, open("data/y_test.pkl", "wb"))
N_CLASSES = 3
EARLY_STOPPING=300
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': N_CLASSES,
    'metric': 'multi_logloss',
    'max_depth': 4,
    'num_leaves': 32,
    'feature_fraction': 0.1,
    # 'bagging_fraction': 0.8,
    # 'bagging_freq': 5,
    'learning_rate': 0.02,
    'lambda_l1': 0.0,
    'lambda_l2': 0.0,
    'verbose': -1,
    'nthread': 12
}  

print(OOF_NAME, DATA, X.shape, X_test.shape)

KFOLD, SHUF, RS = 5, True, 123
OOF_TRAIN = np.zeros((NTRAIN, N_CLASSES))
OOF_TEST = np.zeros((NTEST, N_CLASSES))
val_score_list = []
kf = StratifiedKFold(n_splits=KFOLD, shuffle=SHUF, random_state=RS)

for i, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print("FOLD #", i, end=' ')
    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[val_idx], y[val_idx]
    
    lgtrain = lgb.Dataset(X_train, y_train)
    lgvalid = lgb.Dataset(X_valid, y_valid)

    modelstart = time.time()
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=1000,
        valid_sets=[lgvalid],
        valid_names=['valid'],
        early_stopping_rounds=EARLY_STOPPING,
        verbose_eval=False
    )
    
    pkl.dump(lgb_clf, open("models/%s_%s.pkl"%(OOF_NAME, i), 'wb'))
    val_preds = lgb_clf.predict(X_valid)
    err = log_loss(y_valid, val_preds)
    test_preds = lgb_clf.predict(X_test, raw_score=True)
    test_preds = softmax(test_preds, axis=1)
    err_test = log_loss(y_test, test_preds)
    OOF_TEST += test_preds
    val_preds = lgb_clf.predict(X_valid, raw_score=True)
    val_preds = softmax(val_preds, axis=1)
    OOF_TRAIN[val_idx] = val_preds

    print('Log Loss (val - test): %.5f - %.5f' % (err, err_test))
    val_score_list.append(err)
    # print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))

OOF_TEST /= KFOLD
# print("Loss folds", val_score_list)
print("Average %.5f - %.5f" % (np.mean(val_score_list), log_loss(y_test, OOF_TEST)))

submission = pd.DataFrame(OOF_TEST, columns=['A', 'B', 'NEITHER'])
submission['ID'] = IDs_test
submission.set_index('ID', inplace=True)
submission.to_csv(f'output/' + OOF_NAME + '.csv')

train_oof = pd.DataFrame(OOF_TRAIN, columns=['A', 'B', 'NEITHER'])
train_oof['ID'] = IDs_train
train_oof.set_index('ID', inplace=True)
train_oof['label'] = Y_train_ones
train_oof.to_csv(f'oof/' + OOF_NAME + '.csv')


