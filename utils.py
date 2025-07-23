## File loading utilities ##

import pandas as pd
import os

def load_csv_file(file_path):
    """Loads a CSV file and returns a DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path, header=None)

## Column names loader ##

def load_column_names(file_path):
    """Loads column names from a text file and returns a list."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Column names file not found: {file_path}")
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

## Fix column mismatch ##

def adjust_column_names(dataset, column_names):
    """Aligns the number of column names to the number of columns in the dataset."""
    if len(column_names) > len(dataset.columns):
        column_names = column_names[:len(dataset.columns)]
    elif len(column_names) < len(dataset.columns):
        column_names.extend(['Unnamed'] * (len(dataset.columns) - len(column_names)))
    dataset.columns = column_names
    return dataset


