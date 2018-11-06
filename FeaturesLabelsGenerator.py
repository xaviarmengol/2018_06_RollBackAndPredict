import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging
from OpportunitiesWithHistory import OpportunitiesWithHistory
from DataFrameDict import DataFrameDict
from copy import copy, deepcopy
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

#TODO: Review if should be done here

with open('conf/stage_to_label.json') as handle:
    STAGE_TO_LABEL = json.loads(handle.read())

with open('conf/opp_columns_names.json') as handle:
    op_col = json.loads(handle.read())

with open('conf/hist_columns_names.json') as handle:
    h_col = json.loads(handle.read())


class FeaturesLabelsGenerator:

    def __init__(self, ops_time_line: OpportunitiesWithHistory,
                 df_op_lines: pd.DataFrame,
                 timedelta_in_months: int,
                 df_changes_history: pd.DataFrame,
                 function_to_filter_df=None,
                 sampling_period_in_months=1):

        self._df_op_lines = df_op_lines

        self._time_delta_in_months = timedelta_in_months
        self._history = df_changes_history.reset_index().set_index('Edit Date').sort_index()
        self._function_to_filter_df = copy(function_to_filter_df)
        self._sampling_period_in_months = sampling_period_in_months

        self._y = DataFrameDict()
        self._X = DataFrameDict()

        self._ops = ops_time_line


    @property
    def y(self):
        """Label DataFrame"""
        return self._y


    @property
    def X(self):
        """Features DataFrame"""
        return self._X


    @property
    def ops(self):
        return self._ops


    # REDEFINIR TODA LA FUNCION:
    # - Dataset debe contener toda la información al inicio: Oportunidades + BU + PLINE

    def _filter_df(self, df):

        df_filtered = self._filter_open_ops(df)
        #df_filtered = self._add_bu(df_filtered)
        #df_filtered = self._add_pline(df_filtered)
        df_filtered = self._apply_filter_to_restrict_ops(df_filtered)

        return df_filtered


    def _filter_open_ops(self, df):
        """ """

        mask = (df['Phase/Sales Stage'].map(STAGE_TO_LABEL) == 1)
        return df[mask]


    def _apply_filter_to_restrict_ops(self, df):
        """Apply filter function that has been provided in instantiation. If does not exist, do not filter"""

        if self._function_to_filter_df is None:
            logging.info('DataFrame NOT filtered')
            return df
        else:
            return self._function_to_filter_df(df)


    def calculate_label(self):

        freq_sampling = str(self._sampling_period_in_months) + 'MS'
        range_dates = pd.date_range(self.ops._df_min_date, self.ops._df_max_date, freq=freq_sampling)
        range_dates = list(range_dates[::-1])

        for date in range_dates:
            try:
                df_label_in_date = self._calculate_stage_in_future(date)

            except KeyError:
                continue

            else:
                self.y[date] = df_label_in_date
                logging.debug('New label calculated:', self.y)


    def _calculate_stage_in_future(self, date):

        date_to_check_stage = date + relativedelta(months=self._time_delta_in_months)

        df_ops_date = self._filter_df(self.ops[date])
        df_ops_date_se_account = df_ops_date.index

        # Checked against future df with no filter, to avoid non found opportunities
        df_ops_date_to_check_stage = self.ops[date_to_check_stage]

        stage_of_se_accounts = df_ops_date_to_check_stage.loc[df_ops_date_se_account, op_col['PHASE']]
        stage_of_se_accounts = stage_of_se_accounts.map(STAGE_TO_LABEL)

        return pd.DataFrame(stage_of_se_accounts)


    def calculate_features(self):

        for date in self.y.list_all_dates:

            self.X[date] = self._calculate_features_date(date)
            logging.info('- Calculated df features in date: {}'.format(date))


    def _calculate_features_date(self, date):

        df_ops_date = self._filter_df(self.ops[date])
        df_features_date = pd.DataFrame()

        df_features_date = self._add_basic_info(df_features_date, df_ops_date, date)

        df_features_date, df_num_upd = self._add_num_upd_until_date(df_features_date, date)
        df_features_date, df_num_upd_w0 = self._add_num_upd_until_date(df_features_date, date, window_in_days=31)
        df_features_date, df_num_upd_w1 = self._add_num_upd_until_date(df_features_date, date, window_in_days=91)
        df_features_date, df_num_upd_w2 = self._add_num_upd_until_date(df_features_date, date, window_in_days=182)

        df_features_date = self._add_num_update_ratio(df_features_date, df_num_upd_w0, df_num_upd, 'ratio_w0_vs_w_')
        df_features_date = self._add_num_update_ratio(df_features_date, df_num_upd_w1, df_num_upd, 'ratio_w1_vs_w_')
        df_features_date = self._add_num_update_ratio(df_features_date, df_num_upd_w2, df_num_upd, 'ratio_w2_vs_w_')

        df_features_date = self._add_last_upd_until_date(df_features_date, date)

        df_features_date = self._add_bu(df_features_date)
        df_features_date = self._add_pline(df_features_date)
        df_features_date = self._add_categorical(df_features_date, df_ops_date)

        return df_features_date



    def _add_basic_info(self, df_features_date, df_ops_date, date):

        df_features_date['amount'] = df_ops_date[op_col['AMOUNT']]
        df_features_date['is_sol'] = df_ops_date[op_col['SOLUTION']].map(lambda x: int(x))
        df_features_date['has_sol_center'] = df_ops_date[op_col['SOLUTION_CENTER']].notnull().map(lambda x: int(x))
        df_features_date['age_at_close_date'] = (df_ops_date[op_col['CLOSE_DATE']] - df_ops_date[op_col['CREATED_DATE']]).dt.days
        df_features_date['close_date_month_abs_1'] = np.sin(np.pi * df_ops_date[op_col['CLOSE_DATE']].dt.month / 6.0) # Month is made continuous
        df_features_date['close_date_month_abs_2'] = np.cos(np.pi * df_ops_date[op_col['CLOSE_DATE']].dt.month / 6.0)
        df_features_date['days_to_close'] = (df_ops_date[op_col['CLOSE_DATE']] - date).dt.days
        df_features_date['age'] = (date - df_ops_date[op_col['CREATED_DATE']]).dt.days

        return df_features_date


    def _add_num_upd_until_date(self, df_features_date, to_date, window_in_days=0):

        df_h = self._history
        df_h = df_h[(df_h.index < to_date)]

        if window_in_days != 0:
            from_date = to_date - timedelta(window_in_days)
            df_h = df_h[df_h.index >= from_date]
            col_sufix = '_w' + str(window_in_days)
        else:
            col_sufix = ''

        df_h = df_h.reset_index()
        df_num_upd = df_h.pivot_table(values=h_col['EDIT_DATE'], index=h_col['SE_REF'], columns=h_col['FIELD'], aggfunc='count', fill_value=0)
        df_num_upd['total'] = df_num_upd.sum(axis=1).replace(0,1)

        df_num_upd_with_col_name = df_num_upd.rename(columns=lambda x: 'num_upd_' + x + col_sufix)
        df_features_date = df_features_date.merge(df_num_upd_with_col_name, how='left', left_index=True, right_index=True)

        df_num_upd_perc = df_num_upd.div(df_num_upd['total'], axis=0)
        df_num_upd_perc_with_col_name = df_num_upd_perc.rename(columns=lambda x: 'perc_upd_' + x + col_sufix)
        df_features_date = df_features_date.merge(df_num_upd_perc_with_col_name, how='left', left_index=True, right_index=True)

        # Other Ratios

        df_features_date['close_vs_amount_ratio'] = df_features_date['num_upd_Close Date'] / df_features_date['num_upd_Amount'].replace(0,1)
        df_features_date['age_vs_num_upd_close'] = df_features_date['age'] / df_features_date['num_upd_Close Date'].replace(0,1)
        df_features_date['age_vs_num_upd_amount'] = df_features_date['age'] / df_features_date['num_upd_Amount'].replace(0, 1)

        return df_features_date, df_num_upd


    def _add_num_update_ratio(self, df_features_date, df_value, df_divided_by, prefix):

        df_ratio = df_value.div(df_divided_by).fillna(0)

        df_ratio_with_col_name = df_ratio.rename(columns=lambda x: prefix + x)
        df_features_date = df_features_date.merge(df_ratio_with_col_name, how='left', left_index=True, right_index=True)

        return df_features_date


    def _add_last_upd_until_date(self, df_features_date, to_date):

        df_h = self._history
        df_h = df_h[(df_h.index < to_date)]

        df_h = df_h.reset_index()
        df_last_upd = df_h.pivot_table(values=h_col['EDIT_DATE'], index=h_col['SE_REF'], columns=h_col['FIELD'], aggfunc=np.max)

        # If there is no update, last update = created date
        df_last_upd = df_last_upd.apply(lambda col: col.combine_first(df_last_upd['Created.']), axis=0)

        df_last_upd = df_last_upd.apply(lambda col: (to_date - col).dt.days, axis=0)

        df_last_upd = df_last_upd.rename(columns=lambda x: 'last_upd_' + x)
        df_features_date = df_features_date.merge(df_last_upd, how='left', left_index=True, right_index=True)

        return df_features_date


    def _add_categorical(self, df_features_date, df_ops_date):

        df_features_date['owner_role'] = df_ops_date[op_col['OWNER_ROLE']]
        df_features_date['ms_acc']     = df_ops_date[op_col['MS']]
        df_features_date['mss']        = df_ops_date[op_col['MSS']]
        df_features_date['phase']      = df_ops_date[op_col['PHASE']]
        df_features_date['op_cat']     = df_ops_date[op_col['OP_CATEGORY']]
        df_features_date['cl1']        = df_ops_date[op_col['CL1']]
        df_features_date['cl2']        = df_ops_date[op_col['CL2']]
        #df_features_date['op_lead']   = df_ops_date[col_name['OP_LEADER']

        df_features_date = pd.get_dummies(df_features_date)

        return df_features_date


    def _add_bu(self, df):

        df_op_lines = self._df_op_lines
        df_op_lines.BU = df_op_lines.BU.fillna('NO')
        df_BU = df_op_lines.pivot_table(values='Line Amount', index='SE Reference', columns='BU', aggfunc=np.sum,
                                        fill_value=0)
        df_return = df.merge(df_BU, how='left', left_index=True, right_index=True)

        return df_return


    def _add_pline(self, df):

        df_op_lines = self._df_op_lines
        df_op_lines.prod_line = df_op_lines.prod_line.fillna('NOACT')
        df_plines = df_op_lines.pivot_table(values='Line Amount', index='SE Reference', columns='prod_line', aggfunc=np.sum,
                                        fill_value=0)
        df_return = df.merge(df_plines, how='left', left_index=True, right_index=True)

        return df_return

    def __len__(self):
        return len(self.X)
