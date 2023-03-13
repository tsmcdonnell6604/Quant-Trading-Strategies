import pandas as pd 
from tqdm import tqdm 
import wrds 
import numpy as np 

class TradingStrategy:
    def __init__(self, predicted_results,days_to_hold,spy_data,conn,options_chain=None):
        self.predicted_results = predicted_results.drop(columns=['actual'])
        self.days_to_hold = days_to_hold 
        self.spy_data = spy_data 
        self.conn = conn 
        self.signals = None 
        self.options_chain = options_chain 

    def create_signals(self):
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

    def backtest(self,asset,direction):
        asset.columns = ['Price']
        dates = self.signals.index
        cur_signal = 0
        cur_pos = 0
        prev_price = 0
        returns = pd.DataFrame(index=dates)
        for i in dates:
            date_signal = self.signals.loc[i, 'signal']
            cur_price = asset.loc[i, 'Price']
            daily_pnl = cur_pos * (cur_price - prev_price)

            if date_signal != cur_signal:
                if date_signal != 0:
                    cur_pos = 1 * date_signal * direction
                    cur_signal = date_signal
                    prev_price = cur_price 
                else:
                    cur_pos, cur_signal = 0, 0
            returns.loc[i, 'Daily PnL'] = daily_pnl 
        
        return returns 
    
    def retrieve_options(self, date):
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

    def backtest_options(self,how_far=2):
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
                    cur_pos, cur_signal = 100, 1
                elif date_signal == -1:
                    cur_pos, cur_signal = -100, -1
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
            
            returns_df.loc[date,['Daily PnL','Delta']] = [daily_pnl, total_delta] 
        
        return returns_df  