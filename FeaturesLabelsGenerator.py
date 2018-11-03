
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging
from OpportunitiesWithHistory import OpportunitiesWithHistory
from DataFrameDict import DataFrameDict
from copy import deepcopy


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class FeaturesLabelsGenerator:

    def __init__(self, ops_time_line: OpportunitiesWithHistory,
                 df_op_lines: pd.DataFrame,
                 timedelta_in_months: int,
                 df_changes_history: pd.DataFrame,
                 function_to_filter_df=None):

        #TODO: Not in init. Should be in class
        self.STAGE_TO_LABEL = {'0 - Closed': 0,
                               '7 - Deliver & Validate': 2,
                               '2 - Identify Customer Strategic Initiatives': 1,
                               '3 - Qualify Opportunity': 1,
                               '4 - Influence and Develop': 1,
                               '5 - Prepare & Bid': 1,
                               '6 - Negotiate to Win': 1}

        self._df_op_lines = df_op_lines

        self._time_delta_in_months = timedelta_in_months
        self._history = df_changes_history.reset_index().set_index('Edit Date').sort_index()
        self._function_to_filter_df = function_to_filter_df

        self._y = DataFrameDict()
        self._X = DataFrameDict()

        self._ops = self._pre_filter_ops(deepcopy(ops_time_line)) # TODO: Is it necessary to copy?
        self._ops_complete = ops_time_line # Not necessary to make copy


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


    def _pre_filter_ops(self, ops):
        ops_filtered = DataFrameDict()

        for date, df in ops.items():

            df_filtered = self._filter_open_ops(df)
            df_filtered = self._add_bu(df_filtered)
            df_filtered = self._add_pline(df_filtered)
            df_filtered = self._apply_filter_to_restrict_ops(df_filtered)

            ops_filtered[date] = df_filtered

        return ops_filtered


    def _filter_open_ops(self, df):

        mask = (df['Phase/Sales Stage'].map(self.STAGE_TO_LABEL) == 1)
        return df[mask]


    def _apply_filter_to_restrict_ops(self, df):
        """Apply filter function that has been provided in instantiation. If does not exist, do not filter"""

        pass
        if self._function_to_filter_df is None:
            logging.info('DataFrame NOT filtered')
            return df
        else:
            return self._function_to_filter_df(df)


    def calculate_label(self):

        for date in self.ops.list_all_dates:
            try:
                df_label_in_date = self._calculate_stage_in_future(date)

            except KeyError:
                continue

            else:
                self.y[date] = df_label_in_date
                logging.debug('New label calculated:', self.y)


    def _calculate_stage_in_future(self, date):

        date_to_check_stage = date + relativedelta(months=self._time_delta_in_months)

        df_ops_date = self.ops[date]
        df_ops_date_se_account = df_ops_date.index

        # Checked against future df with no filter, to avoid non found opportunities
        df_ops_date_to_check_stage = self._ops_complete[date_to_check_stage]

        stage_of_se_accounts = df_ops_date_to_check_stage.loc[df_ops_date_se_account, 'Phase/Sales Stage']
        stage_of_se_accounts = stage_of_se_accounts.map(self.STAGE_TO_LABEL)

        #print("X, y ->", date, date_to_check_stage)

        return pd.DataFrame(stage_of_se_accounts)


    def calculate_features(self):

        for date in self.y.list_all_dates:

            self.X[date] = self._calculate_features_date(date)
            logging.info('- Calculated df features in date: {}'.format(date))


    def _calculate_features_date(self, date):

        df_ops_date = self.ops[date]
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

        df_features_date['amount'] = df_ops_date['Amount']
        df_features_date['is_sol'] = df_ops_date['Is a Solution'].map(lambda x: int(x))
        df_features_date['has_sol_center'] = df_ops_date['Solution Center'].notnull().map(lambda x: int(x))
        df_features_date['age_at_close_date'] = (df_ops_date['Close Date'] - df_ops_date['Created Date']).dt.days
        df_features_date['close_date_month_abs_1'] = np.sin(np.pi*df_ops_date['Close Date'].dt.month/6.0) # Month is made continuous
        df_features_date['close_date_month_abs_2'] = np.cos(np.pi*df_ops_date['Close Date'].dt.month/6.0)
        df_features_date['days_to_close'] = (df_ops_date['Close Date'] - date).dt.days
        df_features_date['age'] = (date - df_ops_date['Created Date']).dt.days

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
        df_num_upd = df_h.pivot_table(values='Edit Date', index='SE Reference', columns='Field / Event', aggfunc='count', fill_value=0)
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
        df_last_upd = df_h.pivot_table(values='Edit Date', index='SE Reference', columns='Field / Event', aggfunc=np.max)

        # If there is no update, last update = created date
        df_last_upd = df_last_upd.apply(lambda col: col.combine_first(df_last_upd['Created.']), axis=0)

        df_last_upd = df_last_upd.apply(lambda col: (to_date - col).dt.days, axis=0)

        df_last_upd = df_last_upd.rename(columns=lambda x: 'last_upd_' + x)
        df_features_date = df_features_date.merge(df_last_upd, how='left', left_index=True, right_index=True)

        return df_features_date


    def _add_categorical(self, df_features_date, df_ops_date):

        df_features_date['owner_role'] = df_ops_date['Owner Role']
        df_features_date['ms_acc']     = df_ops_date['Market Segment']
        df_features_date['mss']       = df_ops_date['Market Sub-Segment']
        df_features_date['phase']      = df_ops_date['Phase/Sales Stage']
        df_features_date['op_cat']     = df_ops_date['Opportunity Category']
        df_features_date['cl1']        = df_ops_date['Classification Level 1']
        df_features_date['cl2']       = df_ops_date['Classification Level 2']
        #df_features_date['op_lead']   = df_ops_date['Opportunity Leader']

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
