from env import get_url
import pandas as pd

def get_zillow_data():
    query = '''
    SELECT prop.*, 
       pred1.logerror, 
       pred1.transactiondate, 
       air.airconditioningdesc, 
       arch.architecturalstyledesc, 
       build.buildingclassdesc, 
       heat.heatingorsystemdesc, 
       landuse.propertylandusedesc, 
       story.storydesc, 
       construct.typeconstructiondesc 
FROM   properties_2017 prop 
       LEFT JOIN predictions_2017 pred1 USING (parcelid) 
       INNER JOIN (SELECT parcelid, 
                          Max(transactiondate) maxtransactiondate 
                   FROM   predictions_2017 
                   GROUP  BY parcelid) pred2 
               ON pred1.parcelid = pred2.parcelid 
                  AND pred1.transactiondate = pred2.maxtransactiondate 
       LEFT JOIN airconditioningtype air USING (airconditioningtypeid) 
       LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid) 
       LEFT JOIN buildingclasstype build USING (buildingclasstypeid) 
       LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid) 
       LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid) 
       LEFT JOIN storytype story USING (storytypeid) 
       LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid) 
WHERE  prop.latitude IS NOT NULL 
	   AND prop.longitude IS NOT NULL; 
    '''

    url = get_url('zillow')

    zillow = pd.read_sql(query, url)
    return zillow

def measure_na_columns(df):
    na_column_df = pd.DataFrame(df.isna().sum(), columns=['num_na_rows'])
    na_column_df['pct_na_rows'] = df.isna().sum() / len(df.index)
    return na_column_df

def measure_na_rows(df):
    na_row_df = pd.DataFrame(df.isna().sum(axis=1).value_counts(sort=False), 
                      columns=['num_rows'])
    na_row_df = na_row_df.reset_index()
    na_row_df = na_row_df.rename(columns={'index': 'num_col_missing'})
    na_row_df['pct_cols_missing'] = na_row_df.num_col_missing / len(df.columns.tolist())
    return na_row_df

def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def remove_columns(df, cols_to_remove):  
    df = df.drop(columns=cols_to_remove)
    return df

def prep_zillow(cols_to_remove=[], prop_required_column=.5, prop_required_row=.75):
    zillow = get_zillow_data()
    zillow = zillow[zillow.propertylandusetypeid
                             .isin([261, 262, 263, 264, 266, 268, 273, 276, 279])
                            ]
    zillow = remove_columns(zillow, cols_to_remove)
    zillow = handle_missing_values(zillow, prop_required_column, prop_required_row)
    return zillow