import pandas as pd
from datetime import timedelta, datetime

import acquire

def wrangle_sales():
    # Get the data
    sales = acquire.get_sales_items_stores()
    # Change sale_date to datetime
    sales.sale_date = pd.to_datetime(sales.sale_date, format='%a, %d %b %Y %H:%M:%S %Z')
    # Make index sale_date
    sales = sales.set_index('sale_date')
    # Create a column for month and day of week
    sales['month'] = sales.index.month
    sales['day_of_week'] = sales.index.dayofweek
    # Create a column called sales_total
    sales['sales_total'] = sales.sale_amount * sales.item_price
    # Create diff column that is current sales - previous sales
    sales['sales_diff'] = sales.sales_total.diff()
    sales.sales_diff = sales.sales_diff.fillna(0)
    
    return sales


def wrangle_german_power():
    power = acquire.get_german_power()
    power.Date = pd.to_datetime(power.Date, format='%Y-%m-%d')
    power = power.set_index('Date')
    power['Month'] = power.index.month
    power['Year'] = power.index.year
    
    return power