import env
import pandas as pd

def get_titanic_data():
    url = env.get_url('titanic_db')

    query = '''
    SELECT * 
    FROM passengers
    '''

    return pd.read_sql(query, url)

def get_iris_data():
    url = env.get_url('iris_db')

    query = '''
    SELECT m.*, s.species_name
    FROM measurements m
    JOIN species s ON s.species_id = m.species_id;
    '''

    return pd.read_sql(query, url)