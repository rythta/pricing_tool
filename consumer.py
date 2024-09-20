import sys
import redis
import re
import time
import os
import pandas as pd
from gen_model import gen_model
from predict import predict
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from webdriver_manager.firefox import GeckoDriverManager
from openai import OpenAI
from datetime import datetime, timedelta
from functools import wraps
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import random

def retry_with_exponential_backoff(max_retries=10, base_delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for retries in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (TimeoutError, HttpError, ConnectionError) as e:
                    if retries == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** retries) + (random.random() * 0.1)
                    print(f"Error occurred: {str(e)}. Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
        return wrapper
    return decorator

client = OpenAI()

def setup_browser():
    firefox_options = Options()
    firefox_options.add_argument("--headless")
    service = Service(GeckoDriverManager().install())
    return webdriver.Firefox(service=service, options=firefox_options)

def process_url(driver, url):
    def extract_price_and_shipping(url):
        driver.get(url)
        script = """
        var callback = arguments[arguments.length - 1];
        function waitForElement(selector, timeout = 5000) {
            return new Promise((resolve, reject) => {
                const intervalTime = 100;
                let totalTime = 0;
                const interval = setInterval(() => {
                    const element = document.querySelector(selector);
                    totalTime += intervalTime;
                    if (element) {
                        clearInterval(interval);
                        resolve(element);
                    } else if (totalTime >= timeout) {
                        clearInterval(interval);
                        reject(new Error("Element not found within the specified timeout"));
                    }
                }, intervalTime);
            });
        }
        Promise.all([
            waitForElement('.x-price-primary'),
            waitForElement('.ux-labels-values--shipping')
        ])
        .then(elements => {
            const [priceElement, shippingElement] = elements;
            callback({
                price: priceElement.textContent,
                shipping: shippingElement.textContent
            });
        })
        .catch(error => callback({error: error.message}));
        """
        return driver.execute_async_script(script)

    def calculate_total_cost(price, shipping):
        price_value = float(re.search(r'[\d,]+(\.\d{2})?', price.replace(',', '')).group().replace(',', '') or 0)
        shipping_value = 0 if 'free' in shipping.lower() else float(re.search(r'[\d,]+(\.\d{2})?', shipping.replace(',', '')).group().replace(',', '') or 0)
        return round(price_value + shipping_value)

    result = extract_price_and_shipping(url)
    if 'error' in result:
        raise Exception(f"Error: {result['error']}")
    return calculate_total_cost(result['price'], result['shipping'])

def get_url(item_url_form, condition_id, days):
    end_date = int(time.time() * 1000)
    start_date = int((datetime.fromtimestamp(end_date/1000) - timedelta(days=days)).timestamp() * 1000)
    return f'EXAMPLE_URL' #Replace with url to search

def process_item(csv, condition, item, identifier):
    condition_id = 1000 if condition == 'NEW' else 3000
    item_url_form = re.sub(r'\s+', '+', item)

    6months = get_url(item_url_form, condition_id, 180)
    3years = get_url(item_url_form, condition_id, 1095)

    try:
        data_df = pd.read_csv(csv)
    except:
        print('csv not found')
        return write_data_to_sheet(condition, item, ['No Data','','','','','','', 6months, 3years], identifier)

    if len(data_df) < 2:
        return write_data_to_sheet(condition, item, ['No Data','','','','','','', 6months, 3years], identifier)

    try:
        data, model, mape, gap, num_data_points = gen_model(csv)
        high, avg, low, combined = predict(model, item)

        recent_dp = data[data['time_period'] == 180].shape[0]

        filtered_df = data[data['url'].notna() & (data['url'] != '')].copy()
        if filtered_df.empty:
            filtered_df = data
        filtered_df['proximity'] = (filtered_df['total'] - avg).abs()
        sorted_df = filtered_df.sort_values(by='proximity')
        top_4_closest_rows = sorted_df.head(15 if recent_dp < 3 else 6)
        top_4_closest_rows = top_4_closest_rows.drop_duplicates(subset='title')
        title_string = '\n'.join(top_4_closest_rows['title'])
        top_rows_by_title = top_4_closest_rows.set_index('title').to_dict('index')

        closest_url, listing_price = get_closest_item(item, title_string, top_rows_by_title)

        data = [int(listing_price), closest_url, int(high), int(avg), int(low), int(recent_dp), num_data_points, 6months, 3years]
        write_data_to_sheet(condition, item, data, identifier)
    except Exception as e:
        write_data_to_sheet(condition, item, ['No Data','','','','','','', 6months, 3years], identifier)
        error_type = type(e).__name__
        print(f'Trouble processing item {item}: {error_type} - {e}')

def get_closest_item(item, title_string, top_rows_by_title):
    driver = setup_browser()
    closest_url = 'No Data'  # Initialize with a default value
    listing_price = 0  # Initialize with a default value

    for _ in range(10):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": f"Which of the items below are most similar to this item: {item}? This item does not have a GPU unless explicitly mentioned. Try to find listings of quantity 1. Reply with just one of the similar items below.\n{title_string}"
                }
            ],
            max_tokens=300,
        )
        response_text = response.choices[0].message.content.strip('"\'')
        if response_text == 'NA':
            break
        if response_text in top_rows_by_title:
            db_price = top_rows_by_title[response_text]['total']
            listing_price = db_price
            if isinstance(top_rows_by_title[response_text]['url'], str):
                closest_url = top_rows_by_title[response_text]['url']
                active_price = process_url(driver, closest_url)
                listing_price = min(active_price, db_price)
            return closest_url, listing_price

    return closest_url, listing_price

def write_data_to_sheet(condition, item, data, row):
    append_data_to_sheet(condition, item, data, 'PRICING LOG')
    scopes = ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/spreadsheets"]
    secret_file = os.path.join(os.getcwd(), 'client_secret.json')
    credentials = service_account.Credentials.from_service_account_file(secret_file, scopes=scopes)
    service = build('sheets', 'v4', credentials=credentials)
    spreadsheet_id = 'EXAMPLE_ID' #Replace with spreadsheet ID
    sheet_name = 'PRICING'

    start_cell = f'A{int(row)+1}'
    row_length = 12
    end_column_index = chr(ord(start_cell[0]) + row_length - 1)
    end_cell = f"{end_column_index}{int(start_cell[1:])+1}"
    range_name = f"{sheet_name}!{start_cell}:{end_cell}"

    body = {
        'values': [['FINISHED', condition, item] + data],
        'majorDimension': 'ROWS'
    }

    try:
        service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption='USER_ENTERED',
            body=body
        ).execute()
    except HttpError as error:
        print(f"An error occurred: {error}")
        raise

def append_data_to_sheet(condition, item, data, sheet_name):
    scopes = ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/spreadsheets"]
    secret_file = os.path.join(os.getcwd(), 'client_secret.json')
    credentials = service_account.Credentials.from_service_account_file(secret_file, scopes=scopes)
    service = build('sheets', 'v4', credentials=credentials)
    spreadsheet_id = 'EXAMPLE_ID' #replace with spreadhseet ID

    body = {
        'values': [['FINISHED', condition, item] + data],
        'majorDimension': 'ROWS'
    }

    try:
        return service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range=f"{sheet_name}!A:A",
            valueInputOption='USER_ENTERED',
            insertDataOption='INSERT_ROWS',
            body=body
        ).execute()
    except HttpError as error:
        print(f"An error occurred: {error}")
        raise

def consume_items():
    r = redis.Redis(host='localhost', port=6379, db=0)

    while True:
        item = r.lpop('queue')
        if item is None:
            time.sleep(2)
            continue

        data = item.decode('utf-8')
        print(data)
        csv, condition, item, identifier = data.split(',')
        process_item(csv, condition, item, identifier)

if __name__ == "__main__":
    consume_items()
