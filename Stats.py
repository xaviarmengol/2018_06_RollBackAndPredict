import pandas as pd
import numpy as np

class StatsOps:

    def __init__(self, df_all_dates: pd.DataFrame):
        self.df = df_all_dates
        self.STAGE_TO_LABEL = {'0 - Closed': 0,
                               '7 - Deliver & Validate': 2,
                               '2 - Identify Customer Strategic Initiatives': 1,
                               '3 - Qualify Opportunity': 1,
                               '4 - Influence and Develop': 1,
                               '5 - Prepare & Bid': 1,
                               '6 - Negotiate to Win': 1}

    def open_amount(self):
        #TODO: Weighted?

        df = self.df

        mask = (df['Phase/Sales Stage'].map(self.STAGE_TO_LABEL) == 1) &\
               (df['Opportunity Category'] == 'Simple') &\
               (df['Included in Reporting'] == 'Yes')
        df = df[mask]

        return df['Amount'].groupby([df.date_ts, df['Classification Level 1']]).sum()


    def open_weighted_amount_by_person(self):
        #TODO: Weighted? Use forecated percentage instead?

        df = self.df
        #df.pivot_table(values='', index='', columns='', aggfunc=np.sum)




