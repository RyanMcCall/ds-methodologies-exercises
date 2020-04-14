from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def drop_columns(df):
    return df.drop(columns=[
        'species_id', #same as species_name
        'measurement_id' #same as index
    ])

def rename_columns(df):
    return df.rename(columns = {'species_name': 'species'})

def encode_species(train, test):
    encoder = LabelEncoder()
    encoder.fit(train.species)
    
    train.species = encoder.transform(train.species)
    test.species = encoder.transform(test.species)
    
    return train, test

def prep_iris(df):
    df = drop_columns(df)
    df = rename_columns(df)
    train, test = train_test_split(df, train_size=.7, random_state=13)
    train, test = encode_species(train, test)
    
    return train, test