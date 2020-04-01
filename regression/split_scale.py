import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, MinMaxScaler, RobustScaler

def single_split_my_data(data, train_pct, seed):
    '''Takes in dataframe, training percentage, and a seed and returns train and test'''
    return train_test_split(data, train_size=train_pct, random_state=seed)

def double_split_my_data(X, y, train_pct, seed):
    '''Takes in features, target, training percentage, and a seed and returns train_X, test_X, train_y, test_y'''
    return train_test_split(X, y, train_size=train_pct, random_state=seed)

def make_scaled_dataframe(scaler, data):
    '''Takes a scaler and data and returns a scaled dataframe'''
    return pd.DataFrame(scaler.transform(data), columns=data.columns.values).set_index([data.index.values])

def standard_scaler(train, test):
    scaler = StandardScaler()
    scaler.fit(train)
    train_standard_scaled = make_scaled_dataframe(scaler, train)
    test_standard_scaled = make_scaled_dataframe(scaler, test)
    
    return scaler, train_standard_scaled, test_standard_scaled

def scale_inverse(scaler, train_scaled, test_scaled):
    train_unscaled = pd.DataFrame(scaler.inverse_transform(train_scaled), columns=train_scaled.columns.values).set_index([train_scaled.index.values])
    test_unscaled = pd.DataFrame(scaler.inverse_transform(test_scaled), columns=test_scaled.columns.values).set_index([X_test_scaled.index.values])
    
    return train_unscaled, test_unscaled


def uniform_scaler(train, test):
    scaler = QuantileTransformer(output_distribution='uniform')
    scaler.fit(train)
    train_uniform_scaled = make_scaled_dataframe(scaler, train)
    test_uniform_scaled = make_scaled_dataframe(scaler, test)
    
    return scaler, train_uniform_scaled, test_uniform_scaled

def gaussian_scaler(train, test):
    scaler = PowerTransformer(method='yeo-johnson')
    scaler.fit(train)
    train_gaussian_scaled = make_scaled_dataframe(scaler, train)
    test_gaussian_scaled = make_scaled_dataframe(scaler, test)
    
    return scaler, train_gaussian_scaled, test_gaussian_scaled

def min_max_scaler(train, test):
    scaler = MinMaxScaler()
    scaler.fit(train)
    train_min_max_scaled = make_scaled_dataframe(scaler, train)
    test_min_max_scaled = make_scaled_dataframe(scaler, test)
    
    return scaler, train_min_max_scaled, test_min_max_scaled

def iqr_robust_scaler(train, test):
    scaler = RobustScaler()
    scaler.fit(train)
    train_robust_scaled = make_scaled_dataframe(scaler, train)
    test_robust_scaled = make_scaled_dataframe(scaler, test)
    
    return scaler, train_robust_scaled, test_robust_scaled