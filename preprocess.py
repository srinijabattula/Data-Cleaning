import pandas as pd
import numpy as np
import argparse
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

def load_dataset(file_path):
    if os.path.exists(file_path):
        try:
            dataset = pd.read_csv(file_path, header=None)
            print("Dataset loaded successfully.")
            return dataset
        except Exception as e:
            print(f"Error loading dataset: {e}")
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

def load_column_names(column_names_path):
    if os.path.exists(column_names_path):
        with open(column_names_path, 'r') as file:
            return [line.strip() for line in file.readlines()]
    else:
        raise FileNotFoundError(f"Column names file not found: {column_names_path}")

def assign_column_names(dataset, column_names):
    if len(column_names) > len(dataset.columns):
        column_names = column_names[:len(dataset.columns)]
    elif len(column_names) < len(dataset.columns):
        column_names.extend([f'Unnamed_{i}' for i in range(len(dataset.columns) - len(column_names))])
    dataset.columns = column_names
    return dataset

def handle_missing_data(df):
    print(f"Original shape: {df.shape}")
    df = df.dropna(how='all', axis=1)
    df = df.dropna()
    print(f"Shape after removing missing values: {df.shape}")
    return df

def impute_and_scale(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def create_new_features(df):
    try:
        df['velocity_difference'] = df['20%-perc. pairwise velocity'] - df['50%-perc. pairwise velocity']
        df['velocity_x_duration'] = df['start $x$'] * df['stroke duration']
        df['trajectory_distance_ratio'] = df['length of trajectory'] / df['direct end-to-end distance']
    except KeyError as e:
        print(f"Feature engineering skipped: missing column {e}")
    return df

def encode_categorical(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoded_df = pd.DataFrame(encoder.fit_transform(df[categorical_cols]))
        encoded_df.index = df.index
        df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
    return df

def apply_pca(df, n_components=10):
    pca = PCA(n_components=min(n_components, df.shape[1]))
    components = pca.fit_transform(df)
    pca_df = pd.DataFrame(components, columns=[f'PC{i+1}' for i in range(components.shape[1])])
    return pd.concat([df, pca_df], axis=1)

def main(input_path, column_names_path, output_path):
    df = load_dataset(input_path)
    column_names = load_column_names(column_names_path)
    df = assign_column_names(df, column_names)
    df = handle_missing_data(df)
    df = impute_and_scale(df)
    df = create_new_features(df)
    df = encode_categorical(df)
    df = apply_pca(df, n_components=10)
    df.to_csv(output_path, index=False)
    print(f"Processed dataset saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing Pipeline")
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--columns', required=True, help='Path to column names text file')
    parser.add_argument('--output', required=True, help='Path to save the cleaned CSV file')
    args = parser.parse_args()

    main(args.input, args.columns, args.output)

