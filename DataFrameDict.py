import pandas as pd
import datetime
from datetime import datetime
import bisect
import copy


class DataFrameDict:
    """ Augmented Pandas DataFrame dictionary. The Key is a date, and value is a DataFrame

    This class works as a DataFrames dictionary, with some additional features:
    - Keys (dates) are sorted by date
    - Elements can be accessed by key (date) like a dictionary, or by integer index/slice as a list.
    - When an element (DataFrame) is accessed, it contains as columns, the union of columns of every element

    Examples:
        >> df_dict = DataFrameDict()
        >> date = datetime(2018,1,1)
        >> df_dict[date] = df
        >> df_aux = df_dict[date] # df_aux equals df
        >> df_aux2 = df_dict[0] # df_aux2 equals df
    """


    def __init__(self, date=None, df=None, col_date_name='date_ts'):
        """ Initialize the dictionary with first DataFrame and date.

        Args:
            df: Pandas Dataframe with arbitrary columns
            date: DateTime
            col_date_name: Internal name. If collides with any columns name, an error will be raised.
        """

        self._df_all_dates = pd.DataFrame()
        self._list_all_dates = []

        self._col_date_name = col_date_name

        self._check_and_add_new_df(date, df)


    @property
    def df_all_dates(self):
        """Complete DataFrame with all the DataFrames (elements). Includes an auxiliar column with the date (Key)"""
        return self._df_all_dates

    @property
    def df_all_dates_values(self):
        """Complete DataFrame with all the DataFrames (elements). Does not include the date column (Key)"""
        return self._df_all_dates.drop(columns=[self._col_date_name])

    @property
    def list_all_dates(self):
        """List of all dates (keys), sorted chronologically"""
        return self._list_all_dates

    @property
    def first_date(self):
        return self._list_all_dates[0]

    @property
    def last_date(self):
        return self._list_all_dates[-1]


    def _check_and_add_new_df(self, date, df):
        """ Check if df and date are valid, and add a new DataFrame to the dictionary """

        if self._parameters_have_right_type(date, df):

            self._verify_do_not_contains_col_date_name(df)
            self._verify_date_is_not_assigned(date)
            self._add_new_df(date, df)

        else:
            pass # In case of default values, do nothing


    def _parameters_have_right_type(self, date, df):
        """ Check if parameters have the right type or default value. Otherwise rise an error"""

        if isinstance(df, pd.DataFrame) and isinstance(date, datetime):
            parameters_ok = True

        elif df is None and date is None:
            parameters_ok = False

        else:
            raise TypeError('Parameters types not expected:', type(df), type(date))

        return parameters_ok


    def _verify_do_not_contains_col_date_name(self, df):
        if self._col_date_name in df.columns:
            raise ValueError('DataFrame columns can not contain col_date_name:', self._col_date_name)


    def _verify_date_is_not_assigned(self, date):
        if date in self._list_all_dates:
            raise KeyError('DataFrame already assigned to date:', date)


    def _verify_date_exists(self, date):
        if date not in self._list_all_dates:
            raise KeyError('DataFrame in date requested not found:', date)


    def _add_new_df(self, date, df):
        """ Add a checked df/date into the dictionary """

        df_aux = df.copy(deep=True)
        df_aux[self._col_date_name] = date

        # Add the DataFrame df to the complete DataFrame, and create any new column that doesn't yet exist
        self._df_all_dates = pd.concat([self._df_all_dates, df_aux])

        # Insert the new date in the dates list, keeping the order
        bisect.insort(self._list_all_dates, date)


    def _get_df_in_date(self, date):
        """ Get the DataFrame that key = date, if date exists"""

        self._verify_date_exists(date)
        return self._get_df_in_existing_date(date)


    def _get_df_in_existing_date(self, date):
        """ Get the DataFrame that key = date"""

        col_date_name = self._col_date_name
        mask = self._df_all_dates[col_date_name] == date
        df = self._df_all_dates[mask]
        return df.drop(col_date_name, axis='columns')


    def _delete_date(self, date):
        """ Delete the DataFrame with key=date"""

        del self._list_all_dates[self._list_all_dates.index(date)]

        col_date_name = self._col_date_name
        self._df_all_dates = self._df_all_dates[self._df_all_dates[col_date_name] != date]


    def __len__(self):
        """len(obj): Number of DataFrames stored"""
        return len(self._list_all_dates)


    def __contains__(self, date):
        """date in obj: True if DataFrame in date is stored"""
        return date in self._list_all_dates


    def __getitem__(self, input_item):
        """obj[key]: Get element date, a integer or slice

        In case of date d, returns the DataFrame corresponding to the date d.
        In case of int i, returns the DataFrame number #i in chronological order (being 0 the first).
        In case of slice, returns a NEW DataFrameDict object containing the sliced indexes
        """

        if isinstance(input_item, slice):

            #len_self = len(self._list_all_dates)

            ## First Way
            # def to_pos(value, len_total):
            #     if value < 0:
            #         value_pos = value+len_total
            #     else:
            #         value_pos = value
            #     return value_pos
            #
            # start = to_pos(input_item.start or 0, len_self)
            # stop = to_pos(input_item.stop or len_self, len_self)
            # step = to_pos(input_item.step or 1, len_self)

            ## Second way: start, stop, step = input_item.indices(len_self)

            sliced_object = DataFrameDict()

            # for slice_index in range(start, stop, step):
            #     date_index = self._list_all_dates[slice_index]
            #     df_index = self[date_index]
            #     sliced_object[date_index] = df_index

            for index_slice in self._list_all_dates[input_item]:
                date_slice = self._list_all_dates[index_slice]
                sliced_object[date_slice] = self[date_slice]

            return sliced_object

        elif isinstance(input_item, int):
            date = self._list_all_dates[input_item]

        elif isinstance(input_item, datetime):
            date = input_item

        else:
            raise TypeError('Index has no valid type:', input_item, type(input_item))

        return self._get_df_in_date(date)


    def __setitem__(self, date, df):
        """obj[date] = df"""
        self._check_and_add_new_df(date=date, df=df)


    def __delitem__(self, input_item):
        if isinstance(input_item, int):
            date = self._list_all_dates[input_item]
        else:
            date = input_item

        return self._delete_date(date)


    def __iter__(self):

        return iter(self._list_all_dates)


    #     self._iteration = 0
    #     return self
    #
    #
    # def __next__(self):
    #     """Iterator that return DataFrames (elements), not the date (key)"""
    #
    #     if self._iteration < len(self):
    #         df = self[self._iteration]
    #         return_value = df
    #         self._iteration += 1
    #         return return_value
    #
    #     else:
    #         raise StopIteration


    def items(self):
        """Iterator that return both date, DataFrame"""

        for date in self._list_all_dates:
            yield date, self[date]


    def __repr__(self):
        representation = ''
        for date in self._list_all_dates:
            representation += 'Date: ' + str(date) + '-> DataFrame Shape:' + str(self[date].shape) + '\n'

        return representation


    def __add__(self, other):

        return_obj = copy.deepcopy(self)

        for date_other, df_other in other.items():
            return_obj._check_and_add_new_df(date_other, df_other)

        return return_obj


    def __iadd__(self, other):
        return self + other


