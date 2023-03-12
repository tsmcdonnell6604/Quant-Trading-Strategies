import pandas as pd
import numpy as np 
from tqdm import tqdm 
from statsmodels.tsa.arima.model import ARIMA

class Model:
    def __init__(self, predictors, window_size):
        self.predictors = predictors 
        self.window_size = window_size
        self.predicted_results = None 
        self.betas = None 
        self.pvalues = None 
        self.metrics = None 

    def train_model(self):
        start, end = 0, self.window_size
        columns = self.predictors.iloc[:,1:].columns.to_list()
        columns.append('AR')
        columns.append('MA')
        columns.append('Sigma2')

        predicted_results = pd.DataFrame(index=self.predictors.iloc[self.window_size:].index,columns=['predictions'])
        betas = pd.DataFrame(index=self.predictors.iloc[self.window_size:].index,columns=columns)
        pvalues = pd.DataFrame(index=self.predictors.iloc[self.window_size:].index,columns=columns)

        for i in tqdm(range(len(self.predictors.iloc[126:]))):
            model = ARIMA(self.predictors.iloc[start:end]['VIX Returns'], order=(1,1,1), exog=self.predictors.iloc[start:end,1:]).fit()
            forecast = model.forecast(1, exog=self.predictors.iloc[end,1:])

            start, end = start + 1, end + 1  
            predicted_results.iloc[i] = forecast
            pvalues.iloc[i] = model.pvalues
            betas.iloc[i] = model.params

        predicted_results = predicted_results.dropna()
        predicted_results['actual'] = self.predictors.iloc[self.window_size:,0]
        predicted_results['predictions'] = predicted_results['predictions'].astype(float)

        self.predicted_results = predicted_results 
        self.pvalues = pvalues 
        self.betas = betas 

    def performance_metrics(self, upper_quantile,lower_quantile):
        self.predicted_results['upper threshold'] = self.predicted_results['actual'].rolling(self.window_size - 1).quantile(upper_quantile).shift()
        self.predicted_results['lower threshold'] = self.predicted_results['actual'].rolling(self.window_size - 1).quantile(lower_quantile).shift()

        metrics = pd.DataFrame(index=['Metrics'],columns=['MSE','DA','Up Acc','Down Acc','Prediction Num'])
        mse = np.mean((self.predicted_results['actual'] - self.predicted_results['predictions'])**2)
        da = (np.sign(self.predicted_results['predictions']) == np.sign(self.predicted_results['actual'])).sum() / len(self.predicted_results)

        sig_up = self.predicted_results[self.predicted_results['predictions'] >= self.predicted_results['upper threshold']]
        sig_down = self.predicted_results[self.predicted_results['predictions'] <= self.predicted_results['lower threshold']]

        acc_up = (np.sign(sig_up['predictions']) == np.sign(sig_up['actual'])).sum() / len(sig_up)
        acc_down = (np.sign(sig_down['predictions']) == np.sign(sig_down['actual'])).sum() / len(sig_down)

        metrics.loc['Metrics'] = [mse, da, acc_up, acc_down, len(sig_up) + len(sig_down)]
        self.metrics = metrics 

