import pandas as pd
import numpy as np

#Feature creation involving Lags and Rolling Stats
def engineer_process_features(df):

    airflow_cols = [f"flotation_column_0{i}_air_flow_mean" for i in range(1, 8)]
    airflow_stds = [f"flotation_column_0{i}_air_flow_std" for i in range(1, 8)]
    level_cols = [f"flotation_column_0{i}_level_mean" for i in range(1, 8)]
    level_stds = [f"flotation_column_0{i}_level_std" for i in range(1, 8)]

    df["airflow_total_mean"] = df[airflow_cols].mean(axis=1)
    df["airflow_total_std"] = df[airflow_stds].mean(axis=1)
    df["level_total_mean"] = df[level_cols].mean(axis=1)
    df["level_total_std"] = df[level_stds].mean(axis=1)

    variables_to_process = [
        'pct_iron_feed_mean', 'pct_silica_feed_mean',
        'starch_flow_mean', 'amina_flow_mean',
        'ore_pulp_ph_mean', 'airflow_total_mean',
        'pct_iron_concentrate_mean', 'pct_iron_concentrate_std'
    ]

    for col in variables_to_process:
        if col in df.columns:

            df[f'{col}_lag1'] = df[col].shift(1)
            df[f'{col}_lag3'] = df[col].shift(3)

            df[f'{col}_rolling_mean_3h'] = df[col].shift(1).rolling(window=3).mean()
            df[f'{col}_rolling_std_3h'] = df[col].shift(1).rolling(window=3).std()

    return df


#Extracts cyclic time components from the index.
def add_time_features(df):

    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df["month"] = df.index.month
    df["time_idx"] = np.arange(len(df))

    return df
