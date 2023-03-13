import quandl 
import functools 
import pandas as pd 
import numpy as np
import wrds 

class DataCollection:
    def __init__(self, key, start_date, end_date, years, month_codes):
        self.key = key 
        self.start_date = start_date 
        self.end_date = end_date 
        self.years = years 
        self.month_codes = month_codes 

    @functools.lru_cache()
    def fetch_quandl(self, asset):
        '''
        Retrieves Quandl data 

        Parameters:
        asset (string): Asset to be retrieved
        begin_date (string): Start date of data retrieval
        end_date (string): End date of data retrieval 

        Returns:
        qdata (DataFrame): DataFrame of asset data 
        '''
        qdata = quandl.get(asset,
                        start_date = self.start_date,
                        end_date = self.end_date,
                        api_key = self.key,
                        paginate=True)
        return qdata
    
    @functools.lru_cache()
    def fetch_quandl_table(self,table, ticker):
        '''
        Retrieves Quandl table data 

        Parameters:
        table (string): The table to retrieve 
        ticker (string): The ticker from the table 
        begin_date (string): Start date of data retrieval
        end_date (string): End date of data retrieval 

        Returns:
        qdata (DataFrame): DataFrame of asset data 
        '''
        qdata = quandl.get_table(table,
                            date = { 'gte': self.start_date, 'lte': self.end_date },
                            qopts = {"columns":["date", "close","volume","adj_close"]},
                            ticker = ticker,
                            api_key = self.key,
                            paginate=True)
        qdata = qdata.set_index('date').sort_index(ascending=True)
        return qdata
    
    def get_contract(self,name): 
        '''
        Obtains future contract price information given the month and year codes 

        Parameters:
            name (string): Name of the contract assuming OWF database 
            years (list): List of years of contract expiry
            month_codes (list): List of month codes of contract expiry 
            start (string): The start date of price data (default 2020-21-03)
            end (string): The end date of price data (default 2022-08-31)
            multiplier (float): Multiplier that is applied to the future price (default 1)

        Returns:
            df_filtered (df): Dataframe consisting of future price time-series with desired multiplier applied 
        '''
        df = pd.DataFrame()
        for y in self.years:
            for m in self.month_codes:
                code = 'OWF/' + name + '_' + m + y + '_IVM'
                qdata = quandl.get(code, returns='pandas', api_key=self.key, start_date=self.start_date, end_date=self.end_date)
                df = pd.concat([df, qdata.reset_index()])
        df_filtered = df.sort_values(by=['Date','DtT'])
        df_filtered = df_filtered[df_filtered.DtT >= 30].groupby('Date').first()
        df_filtered = df_filtered[['Future']].rename(columns={'Future':'Price'})
        return df_filtered
    
    def convert_calendar(self,calendar,returns):
        calendar['DATE'] = pd.to_datetime(calendar['Start']).dt.date
        calendar = calendar.set_index('DATE')

        idx = returns.index 

        temp = returns.copy()
        temp.columns = ['VIX']
        temp = temp.reindex(idx, fill_value=None)
        temp['Event'] = np.where(temp.index.isin(calendar.index.values), True, False)
        temp['Days until Event'] = np.where(temp['Event'] == True, 0, returns.groupby((temp['Event'] == True).cumsum()).cumcount(ascending=False)+1)
        temp = temp.dropna(subset=['VIX'])
        temp = temp[['Days until Event']]
        return temp 
    