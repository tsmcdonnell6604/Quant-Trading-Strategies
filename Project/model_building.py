import pandas as pd
import numpy as np 
from tqdm import tqdm 
from statsmodels.tsa.arima.model import ARIMA
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class Model:
    def __init__(self, predictors, window_size):
        self.predictors = predictors 
        self.window_size = window_size
        self.predicted_results = None 
        self.betas = None 
        self.pvalues = None 
        self.metrics = None 

    def train_model(self):
        '''
        Trains the ARIMA(1,1,1) model and sets the class variables 
        '''
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

    def performance_metrics(self,upper_quantile,lower_quantile,static=False):
        '''
        Obtains performance metrics for the ARIMA model

        Parameters:
            upper_quantile (float): Rolling upper quantile
            lower_quantile (float): Rolling lower quantile
            static (bool): Whether to use a rolling quantile 

        Returns:
            Sets the metrics member variable 
        '''
        if static == False:
            self.predicted_results['upper threshold'] = self.predicted_results['actual'].rolling(self.window_size - 1).quantile(upper_quantile).shift()
            self.predicted_results['lower threshold'] = self.predicted_results['actual'].rolling(self.window_size - 1).quantile(lower_quantile).shift()
        else:
            self.predicted_resutls['upper threshold'] = upper_quantile 
            self.predicted_results['lower threshold'] = lower_quantile 

        metrics = pd.DataFrame(index=['Metrics'],columns=['MSE','DA','Up Acc','Down Acc','Prediction Num'])
        mse = np.mean((self.predicted_results['actual'] - self.predicted_results['predictions'])**2)
        da = (np.sign(self.predicted_results['predictions']) == np.sign(self.predicted_results['actual'])).sum() / len(self.predicted_results)

        sig_up = self.predicted_results[self.predicted_results['predictions'] >= self.predicted_results['upper threshold']]
        sig_down = self.predicted_results[self.predicted_results['predictions'] <= self.predicted_results['lower threshold']]

        acc_up = (np.sign(sig_up['predictions']) == np.sign(sig_up['actual'])).sum() / len(sig_up)
        acc_down = (np.sign(sig_down['predictions']) == np.sign(sig_down['actual'])).sum() / len(sig_down)

        metrics.loc['Metrics'] = [mse, da, acc_up, acc_down, len(sig_up) + len(sig_down)]
        self.metrics = metrics 

    def plot_pvalues(self):
        '''
        Plots the pvalues after training the model 
        '''
        titles = tuple(self.pvalues.columns[:-1].to_list())
        fig = make_subplots(
            rows = 3,
            cols = 2,
            subplot_titles=titles 
            )
        
        fig.add_trace(go.Scatter(x=self.pvalues.index,name='SPY',y=self.pvalues['SPY Returns']),row=1,col=1)
        fig.add_trace(go.Scatter(x=self.pvalues.index,name='EAFE',y=self.pvalues['EAFE Returns']),row=1,col=2)
        fig.add_trace(go.Scatter(x=self.pvalues.index,name='Calendar',y=self.pvalues['Days until Event']),row=2,col=1)
        fig.add_trace(go.Scatter(x=self.pvalues.index,name='Volume',y=self.pvalues['Volume Change']),row=2,col=2)
        fig.add_trace(go.Scatter(x=self.pvalues.index,name='AR',y=self.pvalues['AR']),row=3,col=1)
        fig.add_trace(go.Scatter(x=self.pvalues.index,name='MA',y=self.pvalues['MA']),row=3,col=2)

        fig.update_xaxes(title='Date',row=1,col=1)
        fig.update_xaxes(title='Date',row=1,col=2)
        fig.update_xaxes(title='Date',row=2,col=1)
        fig.update_xaxes(title='Date',row=2,col=2)
        fig.update_xaxes(title='Date',row=3,col=1)
        fig.update_xaxes(title='Date',row=3,col=2)

        fig.update_yaxes(title='p-value',row=1,col=1)
        fig.update_yaxes(title='p-value',row=1,col=2)
        fig.update_yaxes(title='p-value',row=2,col=1)
        fig.update_yaxes(title='p-value',row=2,col=2)
        fig.update_yaxes(title='p-value',row=3,col=1)
        fig.update_yaxes(title='p-value',row=3,col=2)

        fig.layout.update(showlegend=False)

        fig.update_layout(
            height=750,
            width=1250,
            margin=dict(t=25, l=25, b=25),
        )

        fig.show()

    def calc_significance(self):
        '''
        Calculates the proportion of significant predictors for each predictor 

        Returns:
            df (DataFrame): DataFrame with the proportion of significant predictors 
        '''
        df = pd.DataFrame(index=self.pvalues.columns[:-1],columns=['Proportion'])
        for predictor in df.index:
            df.loc[predictor] = len(self.pvalues[self.pvalues[predictor] < 0.05]) / len(self.pvalues)
        return df 

    def plot_betas(self):
        '''
        Plots the betas from the ARIMA(1,1,1) model 
        '''
        titles = tuple(self.betas.columns[:-1].to_list())
        fig = make_subplots(
            rows = 3,
            cols = 2,
            subplot_titles=titles 
            )
        
        fig.add_trace(go.Scatter(x=self.betas.index,name='SPY',y=self.betas['SPY Returns']),row=1,col=1)
        fig.add_trace(go.Scatter(x=self.betas.index,name='EAFE',y=self.betas['EAFE Returns']),row=1,col=2)
        fig.add_trace(go.Scatter(x=self.betas.index,name='Calendar',y=self.betas['Days until Event']),row=2,col=1)
        fig.add_trace(go.Scatter(x=self.betas.index,name='Volume',y=self.betas['Volume Change']),row=2,col=2)
        fig.add_trace(go.Scatter(x=self.betas.index,name='AR',y=self.betas['AR']),row=3,col=1)
        fig.add_trace(go.Scatter(x=self.betas.index,name='MA',y=self.betas['MA']),row=3,col=2)

        fig.update_xaxes(title='Date',row=1,col=1)
        fig.update_xaxes(title='Date',row=1,col=2)
        fig.update_xaxes(title='Date',row=2,col=1)
        fig.update_xaxes(title='Date',row=2,col=2)
        fig.update_xaxes(title='Date',row=3,col=1)
        fig.update_xaxes(title='Date',row=3,col=2)

        fig.update_yaxes(title='Beta',row=1,col=1)
        fig.update_yaxes(title='Beta',row=1,col=2)
        fig.update_yaxes(title='Beta',row=2,col=1)
        fig.update_yaxes(title='Beta',row=2,col=2)
        fig.update_yaxes(title='Beta',row=3,col=1)
        fig.update_yaxes(title='Beta',row=3,col=2)

        fig.layout.update(showlegend=False)

        fig.update_layout(
            height=750,
            width=1250,
            margin=dict(t=25, l=25, b=25),
        )

        fig.show()
        
    def plot_predictions(self,start,end):
        '''
        Plots the predictions, actual values, upper, and lower thresholds for a time period 

        Parameters:
            start (string): Start date 
            end (string): End date 
        '''
        fig = make_subplots(rows=1,cols=1,subplot_titles=('ARIMA(1,1,1) Model',))
        period = self.predicted_results.loc[start:end]

        fig.add_trace(go.Scatter(x=period.index,name='Predicted',y=period['predictions']),row=1,col=1)
        fig.add_trace(go.Scatter(x=period.index,name='Actual',y=period['actual']),row=1,col=1)
        fig.add_trace(go.Scatter(x=period.index,name='Upper Threshold',y=period['upper threshold']),row=1,col=1)
        fig.add_trace(go.Scatter(x=period.index,name='Lower Threshold',y=period['lower threshold']),row=1,col=1)

        fig.update_xaxes(title='Date',row=1,col=1)
        fig.update_yaxes(title='VIX Return',row=1,col=1)
        
        fig.update_layout(
            margin=dict(t=25, l=25, b=25),
        )

        fig.show()