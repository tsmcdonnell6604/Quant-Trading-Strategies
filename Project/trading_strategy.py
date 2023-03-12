import pandas as pd 

class TradingStrategy:
    def __init__(self, predicted_results,days_to_hold):
        self.predicted_results = predicted_results.drop(columns=['actual'])
        self.days_to_hold = days_to_hold 
        self.signals = None 

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
                    time_to_last_trade += 1
                if signal == -1:
                    current_position = 'short'
                    time_to_last_trade += 1
            elif current_position == 'long':
                if signal == -1:
                    current_position = 'short'
                    time_to_last_trade = 1
                elif signal == 1:
                    time_to_last_trade = 1
                elif time_to_last_trade >= self.days_to_hold:
                    current_position = 'none'
                    time_to_last_trade = 0 
                else:
                    time_to_last_trade += 1
            elif current_position == 'short':
                if signal == -1:
                    time_to_last_trade = 1
                elif signal == 1:
                    current_position = 'long'
                    time_to_last_trade = 1
                elif time_to_last_trade >= self.days_to_hold:
                    current_position = 'none'
                    time_to_last_trade = 0 
                else:
                    time_to_last_trade += 1
            self.predicted_results.loc[day, 'position'] = current_position 

        signals_df = pd.DataFrame(index=self.predicted_results.index, columns=['signal'])
        signals_df.loc[self.predicted_results['position'] == 'long', 'signal'] = 1
        signals_df.loc[self.predicted_results['position'] == 'short', 'signal'] = -1 
        signals_df.loc[self.predicted_results['position'] == 'none', 'signal'] = 0
        self.signals = signals_df 