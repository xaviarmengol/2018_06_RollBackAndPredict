import logging
import pickle
from FeaturesLabelsGenerator import FeaturesLabelsGenerator


def filter_ops(df):
    mask = (df['Opportunity Category'] == 'Simple') & (df['ID'] >= 50000)
    return df[mask]

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    with open('data/Calculated_Ops_with_Time_line_MS.pkl', 'rb') as f:
        ops, range_dates = pickle.load(f)

    with open('data/Pred_Cifra_csv_in_df.pkl', 'rb') as f:
        df_ops, df_history, df_op_lines = pickle.load(f)


    Xy = FeaturesLabelsGenerator(ops, df_op_lines, timedelta_in_months=6,
                                 df_changes_history=df_history,
                                 function_to_filter_df=filter_ops)

    Xy.calculate_label()
    Xy.calculate_features()

    with open('data/Complete_dataset_MS.pkl', 'wb') as f:
        pickle.dump(Xy, f)

    logging.info('Saved with Pickle')
    #Xy.X.df_all_dates.to_csv('data/all_dates_df_features.csv', sep=';', decimal=',')
