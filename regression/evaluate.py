import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

def zplot_residuals(actual, predicted):
    residuals = actual - predicted

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs[0, 0].hist(residuals, bins=20, ec='black', fc='white')
    axs[0, 0].set(title="Distribution of Residuals")

    axs[0, 1].scatter(actual, predicted, marker='.', c='firebrick')
    axs[0, 1].plot([actual.min(), actual.max()], [actual.min(), actual.max()], ls=':', color='black')
    axs[0, 1].set(title="Actual vs Predicted", xlabel="$y$", ylabel=r"$\hat{y}$")

    axs[1, 0].scatter(actual, residuals, marker='.', c='firebrick')
    axs[1, 0].hlines(0, actual.min(), actual.max(), ls=':', color='black')
    axs[1, 0].set(title="Actual vs Residuals", xlabel="$y$", ylabel=r"$y - \hat{y}$")

    axs[1, 1].scatter(predicted, residuals, marker='.', c='firebrick')
    axs[1, 1].hlines(0, actual.min(), actual.max(), ls=':', color='black')
    axs[1, 1].set(
        title="Predicted vs Residuals", xlabel=r"$\hat{y}$", ylabel=r"$y - \hat{y}$"
    )

    return fig, axs

def plot_actual_vs_predicted(actual, predicted, df):
    plt.scatter(df[actual], df[predicted], marker='.', c='green')
    plt.plot([df[actual].min(), df[actual].max()], [df[actual].min(), df[actual].max()], ls=':', color='black')
    plt.title(f"Actual vs Predicted")
    plt.xlabel(actual)
    plt.ylabel(predicted)

def rplot_residuals(y, yhat, df, baseline=False):
    '''
    y: string; name of target column
    yhat: string; name of prediction column
    df: DataFrame; dataframe that includes previously declared columns
    baseline: optional bool; If true, will return baseline graph also
    '''
    new_df = df.copy()
    
    new_df['residuals'] = new_df[yhat] - new_df[y]

    if baseline:
        df['baseline'] = new_df.y.mean()
        df['baseline_residuals'] = new_df.baseline - new_df[y]

        f, axes = plt.subplots(1, 2, figsize=(8, 4.5))

        sns.scatterplot(x=y, y='baseline_residuals', data=new_df, ax=axes[0])
        sns.lineplot(x=range(round(df[y].min()-1), round(df[y].max())+1), y=0, color='green', ax=axes[0])
        
        sns.scatterplot(x=y, y='residuals', data=new_df, ax=axes[1])
        sns.lineplot(x=range(round(new_df[y].min()-1), round(new_df[y].max())+1), y=0, color='green', ax=axes[1])

        f.tight_layout(pad=2)

    else:
        sns.scatterplot(x=y, y='residuals', data=new_df)
        sns.lineplot(x=range(round(new_df[y].min()-1), round(new_df[y].max())+1), y=0, color='green')

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