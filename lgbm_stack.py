from __future__ import absolute_import, division, print_function
import itertools
import pickle as pkl
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, StratifiedKFold

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

train_df = pd.read_csv('oof/lgbm_large_mgap.csv', index_col='ID')
test_df = pd.read_csv('output/lgbm_large_mgap.csv', index_col='ID')
COL_LABEL = 'label'
OOF_NAME = 'lgbm_stack_ling'
USE_ALL = True

def add_feats(train_fn, test_fn, tag_name, index_col='ID', filetype='csv'):
    if filetype == 'csv':
        df_train = pd.read_csv(train_fn, index_col=index_col)
        df_test = pd.read_csv(test_fn, index_col=index_col)
    elif filetype == 'pkl':
        df_train = pd.read_pickle(train_fn)
        df_test = pd.read_pickle(test_fn)

    for x in df_train.columns:
        if x == COL_LABEL: continue
        train_df[tag_name + '_' + x] = df_train[x]
        test_df[tag_name + '_' + x] = df_test[x]
        
add_feats('oof/lgbm_base_mgap.csv', 'output/lgbm_base_mgap.csv', 
          'lgbm_base_mgap')
add_feats('oof/lgbm_large_gap.csv', 'output/lgbm_large_gap.csv', 
          'lgbm_large_gap')
add_feats('oof/lgbm_base_gap.csv', 'output/lgbm_base_gap.csv', 
          'lgbm_base_gap')

# add_feats('oof/lgbm_large_onames.csv', 'output/lgbm_large_onames.csv', 
#           'lgbm_large_onames')

# add_feats('oof/lgbm_gpt2_gap.csv', 'output/lgbm_gpt2_gap.csv', 
#           'lgbm_gpt2_gap')

# add_feats('oof/nn_large_emb_gap.csv', 'output/nn_large_emb_gap.csv', 
#           'nn_large_emb_gap')
# add_feats('oof/nn_base_emb_gap.csv', 'output/nn_base_emb_gap.csv', 
#           'nn_base_emb_gap')
# add_feats('oof/elmo_emb_gap.csv', 'output/elmo_emb_gap.csv', 
#           'elmo_emb_gap')

# add_feats('oof/xgb_base_mgap.csv', 'output/xgb_base_mgap.csv', 
#           'xgb_base_mgap')
# add_feats('oof/xgb_base_gap.csv', 'output/xgb_base_gap.csv', 
#           'xgb_base_gap')
# add_feats('oof/xgb_large_mgap.csv', 'output/xgb_large_mgap.csv', 
#           'xgb_large_mgap')
# add_feats('oof/xgb_large_gap.csv', 'output/xgb_large_gap.csv', 
#           'xgb_large_gap')

dev_dist_df = pd.read_csv('data/dev_dist.csv', index_col='ID')
val_dist_df = pd.read_csv('data/val_dist.csv', index_col='ID')
test_dist_df = pd.read_csv('data/test_dist.csv', index_col='ID')
train_dist_df = pd.concat([val_dist_df, test_dist_df])
if USE_ALL:
    train_dist_df = pd.concat([val_dist_df, test_dist_df, dev_dist_df])
    
for x in train_dist_df.columns:
    train_df[x] = train_dist_df[x]
    test_df[x] = dev_dist_df[x]

dev_feats_df = pd.read_csv('data/dev_feats.csv', index_col='ID')
val_feats_df = pd.read_csv('data/val_feats.csv', index_col='ID')
test_feats_df = pd.read_csv('data/test_feats.csv', index_col='ID')
train_feats_df = pd.concat([val_feats_df, test_feats_df])
if USE_ALL:
    train_feats_df = pd.concat([val_feats_df, test_feats_df, dev_feats_df])
for x in train_feats_df.columns:
    train_df['feat_' + x] = train_feats_df[x]
    test_df['feat_' + x] = dev_feats_df[x]

predictors = sorted(list(set(train_df.columns) - {COL_LABEL}))
print(OOF_NAME, "- Num Features", len(predictors), len(train_df), len(test_df))
# print(predictors)
X = train_df[predictors].values.astype('float32')
y = train_df[COL_LABEL].astype('int8')

X_test = test_df[predictors].values.astype('float32')
NTRAIN, NTEST = len(train_df), len(test_df)
N_CLASSES = len(set(train_df[COL_LABEL]))
EARLY_STOPPING=250

lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': N_CLASSES,
    'metric': 'multi_logloss',
    'max_depth': 2,
#     'num_leaves': 15,
    'feature_fraction': 0.5,
    # 'bagging_fraction': 0.8,
    # 'bagging_freq': 5,
    'learning_rate': 0.02,
    'lambda_l1': 0.0,
    'verbose': -1,
    'nthread': 12
}  

y_test = pkl.load(open("data/y_test.pkl", "rb"))
KFOLD, SHUF, RS = 5, True, 123
OOF_TRAIN = np.zeros((NTRAIN, N_CLASSES))
OOF_TEST = np.zeros((NTEST, N_CLASSES))
val_score_list = []
kf = StratifiedKFold(n_splits=KFOLD, shuffle=SHUF, random_state=RS)
for i, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print("FOLD #", i, end=' ')
    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[val_idx], y[val_idx]
    
    lgtrain = lgb.Dataset(X_train, y_train, feature_name=predictors)
    lgvalid = lgb.Dataset(X_valid, y_valid, feature_name=predictors)

    modelstart = time.time()
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=1000,
        valid_sets=[lgvalid],
        valid_names=['valid'],
        early_stopping_rounds=EARLY_STOPPING,
        verbose_eval=False,
#         categorical_feature=['D_PA', 'D_PB', 'IN_URL']
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

OOF_TEST /= KFOLD
# print("Loss folds", val_score_list)
print("Average %.5f - %.5f" % ((sum(val_score_list)/KFOLD), log_loss(y_test, OOF_TEST)))

submission = pd.DataFrame(OOF_TEST, columns=['A', 'B', 'NEITHER'])
submission['ID'] = test_df.index
submission.set_index('ID', inplace=True)
submission.to_csv(f'output/' + OOF_NAME + '.csv')

train_oof = pd.DataFrame(OOF_TRAIN, columns=['A', 'B', 'NEITHER'])
train_oof['ID'] = train_df.index
train_oof.set_index('ID', inplace=True)
train_oof['label'] = train_df['label']
train_oof.to_csv(f'oof/' + OOF_NAME + '.csv')
