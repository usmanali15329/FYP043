import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    try:
        data = pd.read_excel(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise

def preprocess_data_chunk(chunk):
    try:
        chunk['imonth'].fillna(1, inplace=True)
        chunk.loc[(chunk['imonth'] <= 0) | (chunk['imonth'] > 12), 'imonth'] = 1
        chunk['iday'].fillna(1, inplace=True)
        chunk.loc[(chunk['iday'] <= 0) | (chunk['iday'] > 31), 'iday'] = 1
        chunk['date'] = pd.to_datetime(chunk[['iyear', 'imonth', 'iday']].astype(str).agg('-'.join, axis=1), errors='coerce')
        chunk.dropna(subset=['date'], inplace=True)
        return chunk
    except Exception as e:
        logging.error(f"Error preprocessing data chunk: {e}")
        raise

def preprocess_data(data, num_threads=None):
    if num_threads is None:
        num_threads = os.cpu_count()  # Set to the number of available CPU cores
    
    chunks = np.array_split(data, num_threads)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(preprocess_data_chunk, chunk) for chunk in chunks]
        processed_chunks = []
        for future in as_completed(futures):
            processed_chunks.append(future.result())
    logging.info("Data preprocessed successfully")
    return pd.concat(processed_chunks)

def omit_empty_columns(data):
    """Omit columns that are completely empty."""
    return data.dropna(axis=1, how='all')

def filter_data_by_date(data, start_date, end_date):
    """Filter data by date range."""
    return data[(data['date'] >= start_date) & (data['date'] <= end_date)]
