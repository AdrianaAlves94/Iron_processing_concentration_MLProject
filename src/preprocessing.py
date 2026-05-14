import pandas as pd

#Performs 'Safe' cleaning: sorting, date conversion, and renaming.

def clean_basic_data(df, date_col='date'):

    # 1. Convert Date to datetime object
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
    else:
        print(f"Warning: {date_col} not found in columns.")

    # 2. Sort by Date
    df = df.sort_values(date_col).reset_index(drop=True)

    # 3. Standardize Column Names
    df.columns = [col.lower().replace(' ', '_').replace('%', 'pct') for col in df.columns]

    # 4. Handle Dtypes
    # Ensures all feature columns are numeric (excluding the date)
    for col in df.columns:
        if col != date_col.lower():
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

#  Splits a sorted DataFrame into training and testing sets. Assumes df is already sorted by time.

def split_data_chronologically(df, train_size=0.8):

    # Calculate the split point
    split_idx = int(len(df) * train_size)

    # Slice the data
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    return train, test