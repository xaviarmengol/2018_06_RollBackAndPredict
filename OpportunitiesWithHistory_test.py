#%run -m pytest
#python -m pytest
#Ctrl Shift F10 (to execute on Pycharm)

import pandas as pd
import pickle
from datetime import datetime, timedelta
import pytest
from OpportunitiesWithHistory import OpportunitiesWithHistory

with open('data/Pred_Cifra_csv_in_df.pkl', 'rb') as f:
    df_ops, df_history, df_op_lines = pickle.load(f)

ops = OpportunitiesWithHistory(df_changes_history=df_history, date=datetime(2018, 6, 28), df=df_ops)


def num_differences_between_df(df1, df2):
    df_test = df1.fillna(0) != df2.fillna(0)
    return df_test.sum().sum()


def test_df_initial_equals_get_df_in_date():

    df_initial = df_ops
    df_get = ops._get_df_in_date(datetime(2018, 6, 28))
    df_getitem = ops[datetime(2018, 6, 28)]
    df_getitem_by_int = ops[0]

    assert num_differences_between_df(df_initial, df_get) == 0
    assert num_differences_between_df(df_initial, df_getitem) == 0
    assert num_differences_between_df(df_initial, df_getitem_by_int) == 0


def test_len_ops():
    assert len(ops) == 1


def test_contains():
    assert datetime(2018, 6, 28) in ops


def test_get_df_in_date():

    df_test_max_date = ops[ops._df_max_date]
    df_test_min_date = ops[ops._df_min_date]

    assert (df_test_max_date.loc['OP-110511-107687', 'Amount'] == 250000.0)
    assert (df_test_min_date.loc['OP-110511-107687', 'Amount'] == 150000.0)


def test_iterator():
    for date in ops:
        assert isinstance(date, datetime)


def test_delete_index():
    _ = ops[datetime(2018, 1, 1)]
    del ops[1]

    len_partial = 0
    for date, df in ops.items():
        len_partial += len(df)

    assert len(ops) == 2
    assert len_partial == len(ops._df_all_dates)


def test_delete_date():
    _ = ops[datetime(2018, 1, 1)]
    del ops[datetime(2018, 1, 1)]

    len_parcial = 0
    for date, df in ops.items():
        len_parcial += len(df)

    assert len(ops) == 2
    assert len_parcial == len(ops._df_all_dates)


def test_get_df_in_date_returns_exception():

    with pytest.raises(KeyError):
        ops._get_df_in_date(ops._df_max_date + timedelta(1))
