import pandas as pd

from os import path

import requests

def make_items():
    # Establish response for max_page
    items_url = 'https://python.zach.lol/api/v1/items'
    response = requests.get(items_url)
    #Get max_page for range in for loop
    max_page = response.json()['payload']['max_page']
    # Create empty data frame for concat
    items = pd.DataFrame(None)
    #Loop through each page and conat on previous pages
    for i in range(1, max_page + 1):
        items_url = f'https://python.zach.lol/api/v1/items?page={i}'
        response = requests.get(items_url)
        item_page = response.json()['payload']['items']
        item_df = pd.DataFrame(item_page)
        items = pd.concat([items, item_df])
    
    items = items.reset_index().drop(columns='index')
    
    items.to_csv('items.csv')
    

def get_items():
    if not path.isfile("items.csv"):
        make_items()
        
    items = pd.read_csv('items.csv', index_col='Unnamed: 0')
    
    return items


def make_stores():
    stores_url = 'https://python.zach.lol/api/v1/stores'
    response = requests.get(stores_url)

    max_page = response.json()['payload']['max_page']

    stores = pd.DataFrame(None)

    for i in range(1, max_page + 1):
        stores_url = f'https://python.zach.lol/api/v1/stores?page={i}'
        response = requests.get(stores_url)
        store_page = response.json()['payload']['stores']
        store_df = pd.DataFrame(store_page)
        
        stores = pd.concat([stores, store_df])

    stores = stores.reset_index().drop(columns='index')
    
    stores.to_csv('stores.csv')
    

def get_stores():
    if not path.isfile("stores.csv"):
        make_stores()
        
    stores = pd.read_csv('stores.csv', index_col='Unnamed: 0')
    
    return stores


def make_sales():
    sales_url = 'https://python.zach.lol/api/v1/sales'
    response = requests.get(sales_url)

    max_page = response.json()['payload']['max_page']

    sales = pd.DataFrame(None)

    for i in range(1, max_page + 1):
        sales_url = f'https://python.zach.lol/api/v1/sales?page={i}'
        response = requests.get(sales_url)
        sale_page = response.json()['payload']['sales']
        sale_df = pd.DataFrame(sale_page)
        
        sales = pd.concat([sales, sale_df])

    sales = sales.reset_index().drop(columns='index')
    
    sales.to_csv('sales.csv')
    

def get_sales(add_items=False, add_stores=False):
    if not path.isfile("sales.csv"):
        make_sales()
        
    sales = pd.read_csv('sales.csv', index_col='Unnamed: 0')
    
    return sales

def get_sales_items_stores():
    items = get_items()
    stores = get_stores()
    sales = get_sales()
    
    data = sales.merge(items, 
                       left_on='item', 
                       right_on='item_id').drop(columns='item_id')
    
    data = data.merge(stores, 
                      left_on='store', 
                      right_on='store_id').drop(columns='store_id')
    
    return data

def get_german_power():
    url = 'https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv'
    power = pd.read_csv(url)
    
    return power