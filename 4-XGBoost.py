import pandas as pd
import numpy as np
from tqdm import trange, tqdm, tqdm_notebook, tqdm_pandas # barra avance
import pickle

#C:\Anaconda3\MinGW\bin

#import os
#mingw_path = r'C:\Anaconda3\MinGW\x86_64-w64-mingw32\bin'
#mingw_path2 = r'C:\Anaconda3\MinGW\bin'
#os.environ['PATH'] = mingw_path + ';' + mingw_path2 + ';' + os.environ['PATH']

import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import ParameterGrid

with open('data/Complete_dataset.pkl', 'rb') as f:
    Xy = pickle.load(f)


best_param_cv = {'colsample_bytree': 0.4,
                 'eta': 0.1,
                 'eval_metric': 'mlogloss',
                 'gamma': 0.2,
                 'max_depth': 10,
                 'min_child_weight': 6,
                 'n_jobs': 4,
                 'num_class': 3,
                 'objective': 'multi:softprob',
                 'random_state': 69,
                 'reg_alpha': 0.1,
                 'scale_pos_weight': 1.0,
                 'silent': 1,
                 'subsample': 0.7}

parameters = {  'colsample_bytree': [0.4],
                'eta': [0.1],
                'eval_metric': ['mlogloss'],
                'gamma': [0.2, 0,4],
                'max_depth': [6, 10],
                'min_child_weight' : [3, 6],
                'n_jobs': [4],
                'num_class': [3],
                'objective': ['multi:softprob'],
                'random_state' : [69],
                'reg_alpha': [0.1, 0.3],
                'scale_pos_weight': [1.0],
                'silent': [1],
                'subsample': [0.7, 0.9]}


param_list = list(ParameterGrid(parameters))
len(param_list)
param_list = list([best_param_cv]) # Comentar para Grid Search

q_samples = len(Xy.X)
q_min_train = 2
q_steps = 2
q_range = q_samples - q_min_train - q_steps

f1_cv_avg_best = 0
f1_cv_list = []


def create_weighs_vector(df_y, percent_value, df_X):
    extra_push = np.zeros(3)
    extra_push[0] = 1
    extra_push[1] = 1
    extra_push[2] = 1

    df_y_return = df_y.values * 0
    for i in range(3):
        df_y_return = df_y_return + (df_y.values == i) * (1 / percent_value[i]) * extra_push[i]
        # df_y_return = df_y_return * df_X['Amount'].values
    return df_y_return


def extract_partial_dataset (Xy, ini, end):

    X = Xy.ops[ini:end].df_all_dates_values
    X_f = Xy.X[ini:end].df_all_dates_values
    y = Xy.y[ini:end].df_all_dates_values

    return X, X_f, y


for param_i, param in enumerate(param_list):

    f1_cv = []
    for i in range(q_range):
        end_train = q_min_train + 1 + i  # no included
        ini_train = 0
        test = (end_train - 1) + q_steps  # 2 quarters, 6 meses para no incluir info del test en el train

        X_train, X_train_f, y_train = extract_partial_dataset(Xy, ini_train, end_train)
        X_test, X_test_f, y_test = extract_partial_dataset(Xy, test, test+1)

        percent_value = pd.value_counts(y_train) / y_train.shape[0]

        w_train = create_weighs_vector(y_train, percent_value, X_train)
        w_test = create_weighs_vector(y_test, percent_value, X_test)

        xg_train = xgb.DMatrix(X_train_f, label=y_train, weight=w_train)
        xg_test = xgb.DMatrix(X_test_f, label=y_test, weight=w_test)

        watchlist = [(xg_train, 'train'), (xg_test, 'test')]

        num_round = 300

        bst = xgb.train(param_list[param_i], xg_train, num_round, watchlist, verbose_eval=False,
                        early_stopping_rounds=20)
        pred_prob = bst.predict(xg_test).reshape(y_test.shape[0], 3)
        pred_label = np.argmax(pred_prob, axis=1)

        f1_score_calc = f1_score(y_test.values, pred_label, average='weighted')
        print("Iter", param_i, "Train:", end_train - 1, "CV:", test, "F1 :", f1_score_calc)

        f1_cv.append(f1_score_calc)

    f1_cv_avg = sum(f1_cv) / len(f1_cv)

    f1_cv_list.append(f1_cv_avg)

    if f1_cv_avg > f1_cv_avg_best:
        f1_cv_avg_best = f1_cv_avg
        best_grid = param
        best_iter_param = param_i

    print("Best", best_iter_param, "F1 CV = ", f1_cv_avg_best)
    print("Iter", param_i, "F1 CV = ", f1_cv_avg)