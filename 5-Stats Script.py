import logging
import pickle
from Stats import StatsOps

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

with open('data/Calculated_Ops_with_Time_line.pkl', 'rb') as f:
    ops, range_dates = pickle.load(f)

a = StatsOps(ops.df_all_dates)
b = a.open_amount()


#TODO: Better computed in notebook