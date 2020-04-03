import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

def plot_residuals(y, yhat, df, baseline=False):
    '''
    y: string; name of target column
    yhat: string; name of prediction column
    df: DataFrame; dataframe that includes previously declared columns
    baseline: optional bool; If true, will return baseline graph also
    '''
    df['residuals'] = df[yhat] - df[y]
    
    if baseline:
        df['baseline'] = df.y.mean()
        df['baseline_residuals'] = df.baseline - df[y]
        
        f, axes = plt.subplots(1, 2, figsize=(8, 4.5))

        sns.scatterplot(x=y, y='baseline_residuals', data=df, ax=axes[0])
        sns.lineplot(x=range(round(df[y].min()-1), round(df[y].max())+1), y=0, color='green', ax=axes[0])

        sns.scatterplot(x=y, y='residuals', data=df, ax=axes[1])
        sns.lineplot(x=range(round(df[y].min()-1), round(df[y].max())+1), y=0, color='green', ax=axes[1])

        f.tight_layout(pad=2)
    
    else:
        sns.scatterplot(x=y, y='residuals', data=df)
        sns.lineplot(x=range(round(df[y].min()-1), round(df[y].max())+1), y=0, color='green')

def regression_errors(y, yhat):
    '''
    y: pd.Series; Series of target values
    yhat: pd.Series; Series of predicted values
    '''
    SSE_yhat = mean_squared_error(y, yhat)*len(y)
    ESS_yhat = sum((yhat - y.mean())**2)
    TSS_yhat = ESS_yhat + SSE_yhat
    MSE_yhat = mean_squared_error(y, yhat)
    RMSE_yhat = sqrt(MSE_yhat)
    
    return SSE_yhat, ESS_yhat, TSS_yhat, MSE_yhat, RMSE_yhat

def baseline_mean_errors(y):
    '''
    y: pd.Series; Series of target values
    '''
    df = pd.DataFrame(y)
    df['baseline'] = df.y.mean()
    
    SSE_baseline = mean_squared_error(df.y, df.baseline)*len(df)
    MSE_baseline = mean_squared_error(df.y, df.baseline)
    RMSE_baseline = sqrt(MSE_baseline)
    
    return SSE_baseline, MSE_baseline, RMSE_baseline

def regression_errors_table(y, yhat, baseline=False):
    '''
    y: pd.Series; Series of target values
    yhat: pd.Series; Series of predict values
    baseline: optional bool; if True, includes baseline on table for comparison
    '''
    error_df = pd.DataFrame(np.array(['SSE', 'ESS', 'TSS', 'MSE', 'RMSE']), columns=['metric'])
    
    if baseline:
        SSE_yhat, ESS_yhat, TSS_yhat, MSE_yhat, RMSE_yhat = regression_errors(y, yhat)
        
        error_df['yhat_values'] = np.array([SSE_yhat, ESS_yhat, TSS_yhat, MSE_yhat, RMSE_yhat])
        
        SSE_baseline, MSE_baseline, RMSE_baseline = baseline_mean_errors(y)
        
        error_df['baseline_values'] = np.array([SSE_baseline, '', '', MSE_baseline, RMSE_baseline])
        
        return error_df
        
    else:
        SSE_yhat, ESS_yhat, TSS_yhat, MSE_yhat, RMSE_yhat = regression_errors(y, yhat)
        
        error_df['yhat_values'] = np.array([SSE_yhat, ESS_yhat, TSS_yhat, MSE_yhat, RMSE_yhat])
                
        return error_df

def better_than_baseline(y, yhat):
    '''
    y: pd.Series; Series of target values
    yhat: pd.Series;
    '''    
    SSE_yhat, ESS_yhat, TSS_yhat, MSE_yhat, RMSE_yhat = regression_errors(y, yhat)
    
    SSE_baseline, MSE_baseline, RMSE_baseline = baseline_mean_errors(y)
    
    return RMSE_yhat < RMSE_baseline

def model_significance(ols_model):
    return ols_model.rsquared, ols_model.f_pvalue