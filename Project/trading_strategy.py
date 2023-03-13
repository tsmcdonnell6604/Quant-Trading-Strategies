import pandas as pd 
from tqdm import tqdm 
import wrds 
import numpy as np 
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class TradingStrategy:
    def __init__(self, predicted_results,days_to_hold,spy_data,vix_data,conn,options_chain=None):
        self.predicted_results = predicted_results.drop(columns=['actual'])
        self.days_to_hold = days_to_hold 
        self.spy_data = spy_data 
        self.vix_data = vix_data 
        self.conn = conn 
        self.signals = None 
        self.options_chain = options_chain 

    def create_signals(self):
        '''
        Creates signals for trading 
        '''
        self.predicted_results['Signal'] = 0
        self.predicted_results.loc[self.predicted_results['predictions'] >= self.predicted_results['upper threshold'], 'Signal'] = 1 
        self.predicted_results.loc[self.predicted_results['predictions'] <= self.predicted_results['lower threshold'], 'Signal'] = -1

        current_position = 'none'
        time_to_last_trade = 0 

        for day in self.predicted_results.index.values:
            signal = self.predicted_results.loc[day]['Signal']
            if current_position == 'none':
                if signal == 1:
                    current_position = 'long'
                elif signal == -1:
                    current_position = 'short'
            elif current_position == 'long':
                if signal == -1:
                    current_position = 'short'
                    time_to_last_trade = 0 
                else:
                    time_to_last_trade += 1 
            elif current_position == 'short':
                if signal == 1:
                    current_position = 'long'
                    time_to_last_trade = 0 
                else:
                    time_to_last_trade +=1 
        
            if time_to_last_trade > self.days_to_hold:
                current_position = 'none'
                time_to_last_trade = 0 

            self.predicted_results.loc[day, 'position'] = current_position 

        signals_df = pd.DataFrame(index=self.predicted_results.index, columns=['signal'])
        signals_df.loc[self.predicted_results['position'] == 'long', 'signal'] = 1
        signals_df.loc[self.predicted_results['position'] == 'short', 'signal'] = -1 
        signals_df.loc[self.predicted_results['position'] == 'none', 'signal'] = 0
        self.signals = signals_df 

    def backtest(self,asset,direction,positions,capital,margin,multiplier=1,mode=0,transaction_fee=0, shorting_fee=0):
        '''
        Backtester for an asset given the signals 

        Parameters:
            asset (DataFrame): Asset to be backtested
            direction (int): Whether it has positive or negative correlation to VIX 
            positions (DataFrame): DataFrame consisting of the positions
            capital (float): Initial capital 
            margin (float): Margin requirement 
            multiplier (int): Multiplier for contracts
            mode (string): spy / future / etf 
            transaction_fee (float): Transaction fee 
            shorting_fee (float): Shorting fee 

        Returns:
            returns (DataFrame): DataFrame of daily PnLs
        '''
        asset.columns = ['Price']
        dates = self.signals.index
        cur_signal = 0
        cur_pos = 0
        prev_price = 0
        returns = pd.DataFrame(index=dates)
        for i in dates:
            date_signal = self.signals.loc[i, 'signal']
            cur_price = asset.loc[i, 'Price']
            daily_pnl = cur_pos * (cur_price - prev_price) * multiplier 

            if date_signal != cur_signal:
                if date_signal != 0:
                    if mode == 'equity':
                        size = 2 * np.round(1.5* positions.loc[i,'Size'] * capital / cur_price)
                        returns.loc[i, 'Transaction Fees'] = size * cur_price * transaction_fee 
                        if cur_pos < 0:
                            returns.loc[i, 'Transaction Fees'] += size * shorting_fee
                    elif mode == 'future':
                        size = np.round(positions.loc[i, 'Size'] * capital / margin)
                        returns.loc[i, 'Transaction Fees'] = size * transaction_fee 
                    elif mode == 'spy':
                        size = np.round(positions.loc[i,'Size'])
                        returns.loc[i, 'Transaction Fees'] = size * cur_price* transaction_fee 
                        if cur_pos < 0:
                            returns.loc[i, 'Transaction Fees'] += size * cur_price * shorting_fee
                    cur_pos = size * date_signal * direction
                    cur_signal = date_signal
                    prev_price = cur_price 
                else:
                    cur_pos, cur_signal = 0, 0
            
            returns.loc[i, 'Daily PnL'] = daily_pnl 
        
        returns['Transaction Fees'] = returns['Transaction Fees'].fillna(0)
        returns['Daily PnL'] -= returns['Transaction Fees']
        return returns 
    
    def retrieve_options(self, date):
        '''
        Retrieves options for a given date using WRDS database 

        Parameters:
            date (string): Date to retrieve options

        Returns:
            df (DataFrame): DataFrame of options chain 
        '''
        query = f"""
        SELECT symbol, date, exdate, cp_flag, best_bid, strike_price,
        best_offer, impl_volatility, delta, gamma, vega, theta 
        FROM optionm_all.opprcd{date[:4]}
        WHERE 1=1
            AND date = '{date}'
            AND symbol like 'SPY%%'
            AND ABS(delta) < 1.0
            AND cp_flag = 'C'
        """
        df = self.conn.raw_sql(query)
        return df

    def obtain_chain(self):
        '''
        Obtains options chain with a given signal set 
        '''
        if self.options_chain.shape[0] != 0:
            print('Options chain already stored. Use TradingStrategy.options_chain')
        else:
            in_pos = False 
            options_chain = pd.DataFrame()

            for date in tqdm(self.signals.index):
                signal = self.signals.loc[date].item()
                date_str = date.strftime('%Y-%m-%d')
                if (in_pos == True) and (signal == 0):
                    options = self.retrieve_options(date_str)
                    options_chain = pd.concat([options_chain, options])
                    in_pos = False 
                elif signal != 0:
                    options = self.retrieve_options(date_str)
                    options_chain = pd.concat([options_chain, options])
                    in_pos = True  
            self.options_chain = options_chain 

    def backtest_options(self,initial_capital,fee=0,how_far=2):
        '''
        Backtester for options 

        Parameters:
            initial_capital (float): Initial capital
            fee (float): Transaction fees
            how_far (int): The expiry date for the option

        Returns:
            returns_df (DataFrame): Options PnL
        '''
        dates = self.signals.index 
        cur_signal = 0
        returns_df = pd.DataFrame(index=dates)
        id, atm_strike = None, None 
        cur_price, prev_price = 0, 0
        cur_pos = 0 
        daily_pnl = 0
        reverse = False
        
        for date in dates:
            date_signal = self.signals.loc[date, 'signal']
            date_str = date.strftime('%Y-%m-%d')
            total_delta = 0
            if date_signal != cur_signal: # Either close or reverse 
                if date_signal != 0:
                    if cur_signal == -1 * date_signal: # to reverse
                        reverse = True 
                        old_id = id 
                        prev_strike = atm_strike 
                        id = None 
                    atm_strike = int(np.floor(self.spy_data.loc[date,'close']))

                options = self.options_chain[(self.options_chain['date'] == date_str) & (self.options_chain['strike_price'] == atm_strike*1000)].sort_values(by='exdate')

                if id:
                    cur_option = options[options['symbol'] == id]
                else:
                    cur_option = options.iloc[[how_far]]
                    delta = cur_option['delta']
                    id = cur_option['symbol'].item()

                cur_price = (1/2) * (cur_option['best_offer'].item() + cur_option['best_bid'].item())

                if reverse == True:
                    prev_option = self.options_chain[(self.options_chain['date'] == date_str) & (self.options_chain['strike_price'] == prev_strike*1000) & (self.options_chain['symbol'] == old_id)].sort_values(by='exdate')
                    last_atm_cur_price = (1/2) * (prev_option['best_offer'].item() + prev_option['best_bid'].item())
                    daily_pnl = cur_pos * (last_atm_cur_price - prev_price)
                    reverse = False 
                    old_id = None 
                else:
                    daily_pnl = cur_pos * (cur_price - prev_price)

                prev_price = cur_price 

                if date_signal == 1: # 1 option for now 
                    cur_pos = 100*np.round(initial_capital/(cur_price*100)) 
                    cur_signal = 1 

                elif date_signal == -1:
                    cur_pos = -100*np.round(initial_capital/(cur_price*100))
                    cur_signal = -1 
                else:
                    cur_pos, cur_signal = 0, 0
                    id = None 
                
                total_delta = np.abs(delta * cur_pos)

            else: # Now do nothing
                if date_signal != 0:
                    options = self.options_chain[(self.options_chain['date'] == date_str) & (self.options_chain['strike_price'] == atm_strike*1000)].sort_values(by='exdate')
                    cur_option = options[(options['symbol'] == id)]
                    cur_price = (cur_option['best_offer'].item() + cur_option['best_bid'].item())/2 
                    daily_pnl = cur_pos * (cur_price - prev_price)
                    prev_price = cur_price 
                else:
                    daily_pnl = 0
            transaction_fee = (fee * abs(cur_pos))/100 
            daily_pnl -= transaction_fee 
            returns_df.loc[date,['Daily PnL','Delta','Transaction Fees']] = [daily_pnl, total_delta,transaction_fee] 
        
        return returns_df  
    
    def position_sizing(self, asset, n):
        '''
        Obtains position sizing for an asset

        Parameters:
            asset (DataFrame): Asset for position sizing
            n (int): n-day correlation

        Returns:
            corr (DataFrame): DataFrame with position sizing 
        '''
        merged = pd.concat([asset.pct_change().dropna(), self.vix_data.pct_change().dropna()],axis=1)
        corr = pd.DataFrame(index=merged.index,columns=['Size'])
        for i in range(n, len(merged)):
            corrs = merged.iloc[i-n:i].corr()
            corr.iloc[i] = corrs.iloc[1,0]
        corr = corr.dropna()
        corr = abs(corr)
        return corr 
    
    def plot_pnls(self, daily_pnls, initial_capital, start, end, title='Total Returns',yaxis='Portfolio Value',adj_h=1200, adj_w=1100):
        '''
        Plot PnLs given daily PnLs 

        Parameters:
            daily_pnls (DataFrame): Daily PnLs 
            initial_capital (float): Initial capital
            start (string): Start date
            end (string): End date 
        '''
        daily_pnls = daily_pnls.loc[start:end]
        cum_pnl = daily_pnls.cumsum()
        portfolios = cum_pnl + initial_capital 

        fig = make_subplots(rows=1,cols=1,subplot_titles=(title,))
        for i in portfolios.columns:
            fig.add_trace(go.Scatter(x=portfolios.index,name=i,y=portfolios[i]),row=1,col=1)

        fig.update_layout(
            height=adj_h,
            width=adj_w,
            margin=dict(t=100, l=50),
            legend_title_text="Asset",
            legend_x=1.005,
            legend_y=0.4,
        )

        fig.update_xaxes(title='Date',row=1,col=1)
        fig.update_yaxes(title=yaxis,row=1,col=1)
        
        fig.update_layout(
            margin=dict(t=25, l=25, b=25),
        )

        fig.show()

    def performance_metrics(self, rets, adj=252):
        '''
        Obtains performance metrics from a PnL set 

        Parameters:
            rets (DataFrame): DataFrame with returns
            adj (int): Adjustment factor

        Returns:
            metrics (DataFrame): DataFrame with performance metrics
        '''
        metrics = pd.DataFrame(columns=rets.columns,index=['Annualized Return',
                                                           'Annualized Volatility',
                                                           'Annualized Sharpe Ratio',
                                                           'Annualized Sortino Ratio',
                                                           'Skewness',
                                                           'Kurtosis',
                                                           'VaR (0.05)',
                                                           'CVaR (0.05)'])
        
        metrics.loc['Annualized Return'] = adj * rets.mean() 
        metrics.loc['Annualized Volatility'] = np.sqrt(adj) * rets.std()
        metrics.loc['Annualized Sharpe Ratio'] = np.sqrt(adj) * rets.mean() / rets.std()
        metrics.loc['Annualized Sortino Ratio'] = np.sqrt(adj) * rets.mean() / rets[rets < 0].std()
        metrics.loc['Skewness'] = rets.skew()
        metrics.loc['Kurtosis'] = rets.kurtosis()
        metrics.loc['VaR (0.05)'] = rets.quantile(0.05)
        metrics.loc['CVaR (0.05)'] = rets[rets <= rets.quantile(0.05)].mean()

        return metrics 
    
    def year_analysis(self,pnls,initial_capital,years):
        '''
        Obtains performance metrics for a series of years

        Parameters:
            pnls (DataFrame): DataFrame of PnLs 
            initial_capital (float): Initial capital for each year 
            years (list): List of years 

        Returns:
            metrics (DataFrame): DataFrame of performance metrics 
        '''
        metrics = pd.DataFrame(columns=years,index=['Annualized Return',
                                                           'Annualized Volatility',
                                                           'Annualized Sharpe Ratio',
                                                           'Annualized Sortino Ratio',
                                                           'Skewness',
                                                           'Kurtosis',
                                                           'VaR (0.05)',
                                                           'CVaR (0.05)'])
        
        for year in years:
            now, next = year, year + 1 
            daily_pnl = pnls.loc[str(now):str(next)]
            port = daily_pnl.cumsum() + initial_capital
            rets = port.pct_change().dropna()

            metrics[year] = self.performance_metrics(rets)

        return metrics 


    def compare_ratios(self,ratios,title,yaxis):
        '''
        Compares two performance ratios 

        Parameters:
            ratios (DataFrame): DataFrame consisting of the ratios
        '''
        trace1 = go.Bar(x=ratios.columns,y=ratios.iloc[0],name=ratios.index[0])
        trace2 = go.Bar(x=ratios.columns,y=ratios.iloc[1],name=ratios.index[1])

        layout = go.Layout(title=title,
                        xaxis={'title': 'Year'},
                        yaxis={'title': yaxis})

        fig = go.Figure(data=[trace1, trace2],layout=layout)
        fig.update_layout(
            margin=dict(t=40, l=25, b=25),
        )

        fig.show()


