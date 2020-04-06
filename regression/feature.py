import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression

def select_kbest(X, y, k):
    '''
    X: pd.DataFrame; Scaled features
    y: pd.DataFrame; Scaled target
    k: int; number of features to return
    
    Returns a list of the column names that are the k best features
    '''
    f_selector = SelectKBest(f_regression, k=k)
    f_selector.fit(X, y)

    f_support = f_selector.get_support()
    f_feature = X.loc[:,f_support].columns
    return f_feature

def do_rfe(X, y, k):
    '''
    X: pd.DataFrame; Scaled features
    y: pd.DataFrame; Scaled target
    k: int; number of features to return
    
    Returns a list of the column names that are the k best features
    '''
    lm = LinearRegression()
    rfe = RFE(lm, k)
    rfe.fit(X, y)

    rfe_features = X.loc[:,rfe.support_].columns
    return rfe_features