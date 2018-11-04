import pandas as pd
import numpy as np
#from tqdm import trange, tqdm, tqdm_notebook, tqdm_pandas # barra avance
import pickle
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import ParameterGrid
import logging
from datetime import datetime, timedelta

from OpportunitiesWithHistory import OpportunitiesWithHistory
from FeaturesLabelsGenerator import FeaturesLabelsGenerator

#from S3_Create_Dataset_Script import filter_ops


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def create_weighs_vector(df_y, percent_value):

    df_y_return = df_y.values * 0
    for i in range(3):
        df_y_return = df_y_return + (df_y.values == i) * (1 / percent_value[i])

    return df_y_return


def extract_partial_dataset (data_set, ini, end):

    X = data_set.ops[ini:end].df_all_dates_values
    X_f = data_set.X[ini:end].df_all_dates_values
    y = data_set.y[ini:end].df_all_dates_values

    return X, X_f, y


def xgboost_dataset (data_set, ini, end, test, parameters, num_round=1000):

    X_train, X_train_f, y_train = extract_partial_dataset(data_set, ini, end)
    X_test, X_test_f, y_test = extract_partial_dataset(data_set, test, test + 1)

    percent_value = pd.value_counts(y_train.iloc[:, 0]) / y_train.shape[0]
    w_train = create_weighs_vector(y_train, percent_value)
    w_test = create_weighs_vector(y_test, percent_value)

    xg_train = xgb.DMatrix(X_train_f, label=y_train, weight=w_train)
    xg_test = xgb.DMatrix(X_test_f, label=y_test, weight=w_test)

    watchlist = [(xg_train, 'train'), (xg_test, 'test')]

    bst = xgb.train(parameters, xg_train, num_round, watchlist, verbose_eval=False,
                    early_stopping_rounds=20)

    predicted_prob = bst.predict(xg_test).reshape(y_test.shape[0], 3)
    predicted_label = np.argmax(predicted_prob, axis=1)

    f1 = f1_score(y_test.values, predicted_label, average='weighted')

    return f1, bst


def xgboost_cv(data_set, param, min_train_periods=6, fixed_train_set=True):

    f1_cv = []
    total_cv_periods = len(data_set) - min_train_periods - MONTHS_PREDICTION

    for cv_index in range(total_cv_periods):

        ini_train = cv_index if fixed_train_set else 0
        end_train = min_train_periods + cv_index
        test = end_train + MONTHS_PREDICTION  # q meses para no incluir info del test en el train

        f1, bst = xgboost_dataset(Xy, ini_train, end_train, test, parameters=param)
        message = "Train from:" + str(ini_train) + "to" + str(end_train - 1) + "CV:", str(test), "F1 :", str(f1)
        logging.info(message)

        f1_cv.append(f1)

    return f1_cv


def xgboost_grid_search_cv(data_set, parameters_list):

    f1_best_parameters = 0
    f1_grid_search = []

    for param_i, param in enumerate(parameters_list):

        f1_cv = xgboost_cv(data_set, param)
        f1_cv_avg = sum(f1_cv) / len(f1_cv)

        f1_grid_search.append(f1_cv_avg)

        if f1_cv_avg > f1_best_parameters:
            f1_best_parameters = f1_cv_avg
            best_grid = param
            best_iter_param = param_i

        print("Best", best_iter_param, "F1 CV = ", f1_best_parameters)
        print("Iter", param_i, "F1 CV = ", f1_cv_avg)

    return f1_grid_search


if __name__ == '__main__':

    #with open('data/Complete_dataset_MS.pkl', 'rb') as f:
    #    Xy = pickle.load(f)

    GRID_SEARCH = False
    MONTHS_PREDICTION = 6
    MIN_TRAIN_PERIODS = 6

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
    parameters_grid = {'colsample_bytree': [0.4],
                       'eta': [0.1],
                       'eval_metric': ['mlogloss'],
                       'gamma': [0.2, 0, 4],
                       'max_depth': [6, 10],
                       'min_child_weight': [3, 6],
                       'n_jobs': [4],
                       'num_class': [3],
                       'objective': ['multi:softprob'],
                       'random_state': [69],
                       'reg_alpha': [0.1, 0.3],
                       'scale_pos_weight': [1.0],
                       'silent': [1],
                       'subsample': [0.7, 0.9]}

    with open('data/Pred_Cifra_csv_in_df.pkl', 'rb') as f:
        df_ops, df_history, df_op_lines = pickle.load(f)

    date_csv = datetime(2018, 6, 28)

    df_op_lines.BU = df_op_lines.BU.fillna('NO')
    df_BU = df_op_lines.pivot_table(values='Line Amount', index='SE Reference', columns='BU', aggfunc=np.sum, fill_value=0)
    df_ops = df_ops.merge(df_BU, how='left', left_index=True, right_index=True)


    df_op_lines.prod_line = df_op_lines.prod_line.fillna('NOACT')
    df_plines = df_op_lines.pivot_table(values='Line Amount', index='SE Reference', columns='prod_line', aggfunc=np.sum, fill_value=0)
    df_ops = df_ops.merge(df_plines, how='left', left_index=True, right_index=True)


    ops = OpportunitiesWithHistory(df_history, date_csv, df_ops)

    def filter_ops(df):
        mask = (df['Opportunity Category'] == 'Simple') & (df['ID'] >= 50000)
        return df[mask]

    Xy = FeaturesLabelsGenerator(ops, df_op_lines, timedelta_in_months=MONTHS_PREDICTION,
                                 df_changes_history=df_history,
                                 function_to_filter_df=filter_ops)
    Xy.calculate_label()
    Xy.calculate_features()

    if GRID_SEARCH:
        param_list = list(ParameterGrid(parameters_grid))
    else:
        param_list = list([best_param_cv])

    logging.info('Parameter search grid over {} parameter set'.format(len(param_list)))

    f1_grid_list = xgboost_grid_search_cv(Xy, param_list)