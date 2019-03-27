from __future__ import absolute_import, division, print_function
import itertools
import pickle as pkl
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, StratifiedKFold

OOF_NAME = 'lgbm'


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
                for lb in ['PA', 'PB']:
                    if len(d[lb]) == 0:
                        break
                    att_tensor = np.array(d[lb])
                    max_att = np.max(att_tensor, axis=0)
                    features.append(max_att[16:20, :].flatten())

                    # mean_att = np.mean(att_tensor, axis=0)
                    # features.append(mean_att[16:20, :].flatten())

                    # min_att = np.min(att_tensor, axis=0)
                    # features.append(min_att[16:20, :].flatten())
                for lb in ['AP', 'BP']:
                    if len(d[lb]) == 0:
                        break
                    att_tensor = np.array(d[lb])
                    max_att = np.max(att_tensor, axis=0)
                    features.append(max_att[14:22, :].flatten())

                    # mean_att = np.mean(att_tensor, axis=0)
                    # features.append(mean_att[16:20, :].flatten())

                    # min_att = np.min(att_tensor, axis=0)
                    # features.append(min_att[16:20, :].flatten())

                if len(d[lb]) == 0:
                    continue
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


IDs_train1, X_train1, Y_train1 = get_data('./data/large-atts-gap-validation.pkl')
IDs_train2, X_train2, Y_train2 = get_data('./data/large-atts-gap-test.pkl')
IDs_test, X_test, Y_test = get_data('./data/large-atts-gap-development.pkl')
IDs_train = IDs_train1 + IDs_train2  # + IDs_test
X_train = X_train1 + X_train2  # + X_test
Y_train = Y_train1 + Y_train2  # + Y_test

NTRAIN, NTEST = len(X_train), len(X_test)
Y_train_ones = convert_label(Y_train)
Y_test_ones = convert_label(Y_test)

X, y = np.array(X_train), np.array(Y_train_ones)
X_test, y_test = np.array(X_test), np.array(Y_test_ones)
print(X.shape, X_test.shape)
N_CLASSES = 3
EARLY_STOPPING = 300
lgbm_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': N_CLASSES,
    'metric': 'multi_logloss',
    # 'max_depth': 15,
    'num_leaves': 127,
    'feature_fraction': 0.2,
    # 'bagging_fraction': 0.8,
    # 'bagging_freq': 5,
    'learning_rate': 0.02,
    'lambda_l1': 0.0,
    'lambda_l2': 0.0,
    'verbose': -1,
    'nthread': 12
}

KFOLD, SHUF, RS = 5, True, 123
OOF_TRAIN = np.zeros((NTRAIN, N_CLASSES))
OOF_TEST = np.zeros((NTEST, N_CLASSES))
val_score_list = []
kf = StratifiedKFold(n_splits=KFOLD, shuffle=SHUF, random_state=RS)
# dtest = xgb.DMatrix(data=X_test)
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

    val_preds = lgb_clf.predict(X_valid)
    err = log_loss(y_valid, val_preds)
    test_preds = lgb_clf.predict(X_test, raw_score=True)
    test_preds = softmax(test_preds, axis=1)
    err_test = log_loss(y_test, test_preds)
    OOF_TEST += test_preds
    val_preds = lgb_clf.predict(X_valid, raw_score=True)
    val_preds = softmax(val_preds, axis=1)
    OOF_TRAIN[val_idx] = val_preds

    print('Log Loss: %.5f - %.5f' % (err, err_test))
    val_score_list.append(err)
    # print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))

OOF_TEST /= KFOLD
print("Loss folds", val_score_list)
# print("Average log loss %.5f" %sum(val_score_list)/KFOLD)
print("Test log loss: %.5f" % log_loss(y_test, OOF_TEST))

submission = pd.DataFrame(OOF_TEST, columns=['A', 'B', 'NEITHER'])
submission['ID'] = IDs_test
submission.set_index('ID', inplace=True)
# submission = pd.read_csv('input/sample_submission_stage_1.csv', index_col='ID')
# for _id, pred in zip(IDs_test, OOF_TEST):
#     submission.loc[_id, 'A'] = pred[0]
#     submission.loc[_id, 'B'] = pred[1]
#     submission.loc[_id, 'NEITHER'] = pred[2]
submission.to_csv(f'output/' + OOF_NAME + '.csv')

# OOF_COLS = [OOF_NAME + str(i) for i in range(N_CLASSES)]
# for i in range(N_CLASSES):
#   col_name = OOF_COLS[i]
#   train_df[col_name] = OOF_TRAIN[:, i]
#   test_df[col_name] = OOF_TEST[:, i]

# train_df[OOF_COLS].to_pickle(OOF_NAME + "_oof_train.pkl")
# test_df[OOF_COLS].to_pickle(OOF_NAME + "_oof_test.pkl")
