import pandas as pd
import datetime
from datetime import datetime
import logging
import bisect
from DataFrameDict import DataFrameDict


class OpportunitiesWithHistory (DataFrameDict):
    """ Extends DataFrameDict class. In case that the requested date doesn't exist, the DataFrame is calculated.

    To calculate the new DataFrame, the class roll back the closest DataFrame in the future to the requested date using
    df_changes_history.
    If new DataFrame can not be calculated, raises an error
    """

    def __init__(self, df_changes_history, date=None, df=None, col_date_name='date_ts'):

        DataFrameDict.__init__(self, date, df)

        self._history = df_changes_history
        self._update_df_min_max_dates()
        self._update_history_min_max_time()


    @property
    def history(self):
        return self._history

    @history.setter
    def history(self, df_changes_history):
        self._history = df_changes_history
        self._update_history_min_max_time()


    def _update_history_min_max_time(self):
        self._history_min_date = self._history.index.min()
        self._history_max_date = self._history.index.max()


    def _update_df_min_max_dates(self):
        # TODO: Review if correct.
        self._df_min_date = self._df_all_dates['Close Date'].min()
        self._df_max_date = self.last_date


    def _get_df_in_date(self, date):
        """ Get the DataFrame that key = date. If doesn't exist but can be calc., is calculated, updated and returned"""

        try:
            df_date = DataFrameDict._get_df_in_date(self, date)

        except KeyError:
            if self._is_date_in_history_range(date) and self._is_date_in_df_range(date):
                logging.debug('New df is calculated in date:', date)
                df_date = self._calculate_df_in_past_date(date)
                self[date] = df_date

            else:
                raise KeyError('Date requested do not exist and can not be calculated:', date)

        return df_date


    def _is_date_in_history_range(self, date):
        return date >= self._history_min_date and date <= self._history_max_date


    def _is_date_in_df_range(self, date):
        return date >= self._df_min_date and date <= self._df_max_date


    def _calculate_df_in_past_date(self, to_date):
        """ Calculate the new DataFrame, rolling back the closest DataFrame that exist in the future

        Args:
            to_date: date of the new DataFrame that we want to calculate

        Returns: calculated DataFrame in date
        """

        date_closest = self._get_closest_existing_date_in_future(to_date)
        df_closest = self[date_closest]

        df_op_rolled_back = df_closest.copy(deep=True)

        # keep only ops that existed in to_date
        df_op_rolled_back = df_op_rolled_back[df_op_rolled_back['Created Date'] <= to_date]

        # keep only change history between dates
        df_history_to_apply = self._history[(self._history.index >= to_date) & (self._history.index <= date_closest)]

        for index, row in df_history_to_apply.iterrows():

            op_se_ref = row['SE Reference']
            op_row_name = row['Field / Event']
            op_old_value = row['Old Value']

            def field_can_be_updated():
                return op_row_name in df_op_rolled_back.columns and op_se_ref in df_op_rolled_back.index

            if field_can_be_updated():

                if op_row_name == 'Amount':
                    op_old_value = self._convert_to_correct_amount(op_old_value)

                if op_row_name == 'Close Date':
                    op_old_value = self._convert_to_correct_datetime(op_old_value)

                df_op_rolled_back.loc[op_se_ref, op_row_name] = op_old_value

        return df_op_rolled_back


    def _get_closest_existing_date_in_future(self, date):
        """ Get closest date in future. Uses that list of dates is always sorted"""

        index_closest_future = bisect.bisect(self._list_all_dates, date)
        return self._list_all_dates[index_closest_future]


    def _convert_to_correct_amount(self, amount_str):
        """ Remove the currency symbol and change the decimals to . based """

        if not (pd.isnull(amount_str)):
            amount = amount_str[4:].replace('.', '').replace(',', '.')
            amount = pd.to_numeric(amount)
        else:
            amount = amount_str
        return amount


    def _convert_to_correct_datetime(self, date_str):
        if not (pd.isnull(date_str)):
            date = datetime.strptime(date_str, '%d/%m/%Y')
        else:
            date = date_str
        return date


    def _delete_date(self, date):
        DataFrameDict._delete_date(self, date)
        self._update_df_min_max_dates()


    def __add__(self, other):
        return_obj = DataFrameDict.__add__(self, other)
        self._update_df_min_max_dates()

        return return_obj

    def __iadd__(self, other):
        return_obj = DataFrameDict.__iadd__(self, other)
        self._update_df_min_max_dates()

        return return_obj

