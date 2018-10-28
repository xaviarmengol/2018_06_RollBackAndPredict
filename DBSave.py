import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from tqdm import trange, tqdm, tqdm_notebook, tqdm_pandas # barra avance
import logging
import pickle
from OpportunitiesWithHistory import OpportunitiesWithHistory
from DataFrameDict import DataFrameDict
from memory_profiler import profile
from sqlalchemy import create_engine

with open('data/Pred_Cifra_csv_in_df.pkl', 'rb') as f:
    df_ops, df_history, df_op_lines = pickle.load(f)

disk_engine = create_engine('sqlite:///data/my_lite_store.db')
df_ops.to_sql('df_ops', disk_engine, if_exists='replace')

#conn = sqlite3.connect("flights.db")
#cur = conn.cursor(