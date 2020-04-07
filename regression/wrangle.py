import pandas as pd
import numpy as np
from env import get_url

def wrangle_telco():
    '''
    Returns a dataframe with the cleaned telco data
    '''
    url = get_url('telco_churn')

    query = '''
    SELECT customer_id, monthly_charges, tenure, total_charges
    FROM customers
    JOIN contract_types USING (contract_type_id)
    WHERE contract_type = 'Two year'
    '''

    df = pd.read_sql(query, url)

    df.replace(' ', np.nan, inplace=True)

    df = df.dropna()

    df.total_charges = df.total_charges.astype('float')

    return df