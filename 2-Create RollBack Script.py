import pandas as pd
from datetime import datetime, timedelta
import logging
import pickle
from OpportunitiesWithHistory import OpportunitiesWithHistory

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

with open('data/Pred_Cifra_csv_in_df.pkl', 'rb') as f:
    df_ops, df_history, df_op_lines = pickle.load(f)

date_csv = datetime(2018, 6, 28)

ops = OpportunitiesWithHistory(df_history, date_csv, df_ops)

range_dates = pd.date_range(ops._df_min_date, ops._df_max_date, freq='QS')
range_dates = list(range_dates[::-1])


for date in range_dates:
    _ = ops[date]
    logging.info('- Date done: {}'.format(date))


with open('data/Calculated_Ops_with_Time_line.pkl', 'wb') as f:
    pickle.dump([ops, range_dates], f)

logging.info('Saved with Pickle')
