import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def iris_drop_columns(df):
    return df.drop(columns=[
        'species_id', #same as species_name
        'measurement_id' #same as index
    ])

def iris_rename_columns(df):
    return df.rename(columns = {'species_name': 'species'})

def iris_encode_species(train, test):
    encoder = LabelEncoder()
    encoder.fit(train.species)
    
    train.species = encoder.transform(train.species)
    test.species = encoder.transform(test.species)
    
    return train, test

def prep_iris(df):
    df = iris_drop_columns(df)
    df = iris_rename_columns(df)
    train, test = train_test_split(df, train_size=.7, random_state=13)
    train, test = iris_encode_species(train, test)
    
    return train, test

def titanic_drop_columns(df):
    return df.drop(columns='deck')

def impute_data(train, test, strategy, column_list):
    imputer = SimpleImputer(strategy=strategy)
    train[column_list] = imputer.fit_transform(train[column_list])
    test[column_list] = imputer.transform(test[column_list])
    return train, test
    
    
def titanic_handle_missing_values(train, test):
    train = train.fillna(np.nan)
    test = test.fillna(np.nan)
    
#     imputer = SimpleImputer(strategy='most_frequent')

#     train.embarked = imputer.fit_transform(train[['embarked']])
#     test.embarked = imputer.fit_transform(train[['embarked']])
    
#     train.embark_town = imputer.fit_transform(train[['embark_town']])
#     test.embark_town = imputer.fit_transform(train[['embark_town']])
    
    train, test = impute_data(train, test, 'most_frequent', ['embarked', 'embark_town'])
    
#     imputer = SimpleImputer(strategy='median')

#     train.age = imputer.fit_transform(train[['age']])
#     test.age = imputer.fit_transform(train[['age']])
    
    train, test = impute_data(train, test, 'median', ['age'])
    
    return train, test
    
def titanic_encode_embarked(train, test):
    encoder = LabelEncoder()
    
    train['embarked_encoded'] = encoder.fit_transform(train.embarked)
    test['embarked_encoded'] = encoder.fit_transform(test.embarked)
    
    return train, test

def titanic_scale_data(train, test):
    scaler = MinMaxScaler()

    scaler.fit(train[['age', 'fare']])

    train_scaled = pd.DataFrame(scaler.transform(train[['age', 'fare']]), 
                                  columns=['age_scaled', 'fare_scaled'],
                                  index=train.index)
    test_scaled = pd.DataFrame(scaler.transform(test[['age', 'fare']]), 
                                  columns=['age_scaled', 'fare_scaled'],
                                  index=test.index)

    train = train.join(train_scaled)
    test = test.join(test_scaled)
    
    return train, test


def prep_titanic(df):
    df = titanic_drop_columns(df)
    train, test = train_test_split(df, train_size=.7, random_state=13)
    train, test = titanic_handle_missing_values(train, test)
    train, test = titanic_encode_embarked(train, test)
    train, test = titanic_scale_data(train, test)
    
    return train, test