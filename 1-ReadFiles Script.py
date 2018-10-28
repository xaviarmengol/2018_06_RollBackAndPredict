import pandas as pd
import logging
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Read all Dataframes

logging.info('DataFrames: Reading...')


def read_csv_bfo_to_df(path):
    df = pd.read_csv(path, sep=',', decimal=',',  encoding="ISO-8859-1", dayfirst=True, low_memory=False)
    df = df.iloc[:-5, :]  # 5 last lines of bfo_csv are comments
    return df


df_ops = read_csv_bfo_to_df('data/Listado_Ops_2015_Hoy_ES_no_loc.csv')
df_history = read_csv_bfo_to_df('data/History_Ops_ES_2015_Today_no_loc.csv')
df_op_lines = read_csv_bfo_to_df('data/Listado_OPLINE_2015_Hoy_ES_no_loc.csv')

logging.info('DataFrames: Read')

# Adapt Opportunities Dataframe

logging.info('Opps DataFrame: Correcting...')


def convert_day_string_series_to_datetime(series):
    return pd.to_datetime(series, format='%d/%m/%Y')


df_ops['Created Date'] = convert_day_string_series_to_datetime(df_ops['Created Date'])
df_ops['Close Date'] = convert_day_string_series_to_datetime(df_ops['Close Date'])
df_ops = df_ops.set_index('SE Reference').sort_index()

logging.info('Opps DataFrame: Corrected')

# Adapt Opp Lines Dataframe

logging.info('Opps Lines DataFrame: Correcting...')


def get_first_chars_of_series(series, num_chars):
    return series.apply(lambda x: x[:num_chars])


df_op_lines['prod_line'] = get_first_chars_of_series(df_op_lines['Product Line'].fillna("NO"), num_chars=5)
df_op_lines['BU'] = get_first_chars_of_series(df_op_lines['Product Line'].fillna("NO"), num_chars=2)

logging.info('Opps Lines DataFrame: Corrected')

# Adapt History Dataframe

logging.info('History DataFrame: Correcting...')


def convert_day_hour_string_series_to_datetime(series):
    return pd.to_datetime(series, format='%d/%m/%Y %H:%M')


df_history['Edit Date'] = convert_day_hour_string_series_to_datetime(df_history['Edit Date'])
df_history['day'] = df_history['Edit Date'].dt.normalize()


def keep_only_first_change_of_each_day(df):
    df_first = df.pivot_table(values='History ID', index=['SE Reference', 'Field / Event', 'day'], aggfunc='min')
    df_first = df_first.reset_index()
    df_only_first_change = df.merge(df_first[['History ID']], how='right', on='History ID')
    return df_only_first_change


df_history = keep_only_first_change_of_each_day(df_history)
df_history = df_history.set_index('Edit Date').sort_index(ascending=False) # Sorted to roll back


logging.info('History DataFrame: Corrected')




with open('data/Pred_Cifra_csv_in_df.pkl', 'wb') as f:
    pickle.dump([df_ops, df_history, df_op_lines], f)

logging.info('Dataframes saved with Pickle')