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

test_df = pd.read_csv('output/lgbm_large_mgap_stage2.csv', index_col='ID')
LING = True
if LING:
    OOF_NAME = 'lgbm_stack_ling'
else:
    OOF_NAME = 'lgbm_stack'

def add_feats(test_fn, tag_name, index_col='ID', filetype='csv'):
    if filetype == 'csv':
        df_test = pd.read_csv(test_fn, index_col=index_col)
    elif filetype == 'pkl':
        df_test = pd.read_pickle(test_fn)

    for x in df_test.columns:
        test_df[tag_name + '_' + x] = df_test[x]
        
add_feats('output/lgbm_base_mgap_stage2.csv', 
          'lgbm_base_mgap')
add_feats('output/lgbm_large_gap_stage2.csv', 
          'lgbm_large_gap')
add_feats('output/lgbm_base_gap_stage2.csv', 
          'lgbm_base_gap')

if LING:
    test2_dist_df = pd.read_csv('data/test2_dist.csv', index_col='ID')
    for x in test2_dist_df.columns:
        test_df[x] = test2_dist_df[x]

    test2_feats_df = pd.read_csv('data/test2_feats.csv', index_col='ID')
    for x in test2_feats_df.columns:
        test_df['feat_' + x] = test2_feats_df[x]

predictors = sorted(list(test_df.columns))
print(OOF_NAME, "- Num Features", len(predictors), len(test_df))
# print(predictors)

X_test = test_df[predictors].values.astype('float32')
NTEST = len(test_df)
N_CLASSES = 3
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
submission['ID'] = test_df.index
submission.set_index('ID', inplace=True)
submission.to_csv(f'output/' + OOF_NAME + '_stage2.csv')