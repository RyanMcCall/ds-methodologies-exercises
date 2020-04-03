import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pydataset import data
from statsmodels.formula.api import ols
from math import sqrt
import split_scale
import warnings
warnings.filterwarnings('ignore')

def plot_residuals(x, y, yhat, df, baseline=False):
    '''
    x: string; name of feature column
    y: string; name of target column
    yhat: string; name of prediction column
    df: DataFrame; dataframe that includes previously declared columns
    baseline: optional string; name of baseline column
    '''
    
    df['residuals'] = df[yhat] - df[y]
    
    if baseline:
        df['baseline_residuals'] = df[baseline] - df[y]
        
        f, axes = plt.subplots(1, 2, figsize=(8, 4.5))

        sns.scatterplot(x=x, y='baseline_residuals', data=df, ax=axes[0])
        sns.lineplot(x=range(round(df[x].min()), round(df[x].max())), y=0, color='green', ax=axes[0])

        sns.scatterplot(x=x, y='residuals', data=train, ax=axes[1])
        sns.lineplot(x=range(round(df[x].min()), round(df[x].max())), y=0, color='green', ax=axes[1])

        f.tight_layout(pad=2)
    
    else:
        sns.scatterplot(x=x, y='residuals', data=train)
        sns.lineplot(x=range(round(df[x].min()), round(df[x].max())), y=0, color='green')