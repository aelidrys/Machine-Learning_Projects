import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from datetime import datetime as dt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, PolynomialFeatures
from geopy.distance import geodesic as GD
from scipy.stats import zscore
from visualize import simple_graph, histogram
from tools import wescr
import argparse



def data_encoding(df, val_df):
    # Data Encoding
    df['store_and_fwd_flag'] = LabelEncoder().fit_transform(df['store_and_fwd_flag'])
    val_df['store_and_fwd_flag'] = LabelEncoder().fit_transform(val_df['store_and_fwd_flag'])
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    val_df['pickup_datetime'] = pd.to_datetime(val_df['pickup_datetime'])
    return df, val_df



def outliers_treatment(df, val_df):
    df['z_score'] = np.abs(zscore(df['trip_duration']))
    df = df[df['z_score'] < 3]  # Remove values where z-score is greater than 3
    df['z_score'] = np.abs(zscore(df['distance']))
    df = df[df['z_score'] < 3]  # Remove values where z-score is greater than 3

    val_df['z_score'] = np.abs(zscore(val_df['trip_duration']))
    val_df = val_df[val_df['z_score'] < 3]  # Remove values where z-score is greater than 3
    val_df['z_score'] = np.abs(zscore(val_df['distance']))
    val_df = val_df[val_df['z_score'] < 3]  # Remove values where z-score is greater than 3
    return df, val_df



def drop_columns(df, val_df):
    column_to_drop = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
                      'Unnamed: 0', 'vendor_id','id' , 'z_score', 'store_and_fwd_flag']
    small_column_to_drop = ['Unnamed: 0', 'vendor_id','id' , 'z_score']
    df = df.drop(columns=small_column_to_drop)
    val_df = val_df.drop(columns=small_column_to_drop)
    return df, val_df



def feature_engineering(df, val_df):
    # Add features
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['pickup_month'] = df['pickup_datetime'].dt.month
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    val_df['pickup_datetime'] = pd.to_datetime(val_df['pickup_datetime'])
    val_df['pickup_month'] = val_df['pickup_datetime'].dt.month
    val_df['pickup_hour'] = val_df['pickup_datetime'].dt.hour

    # Selecting features
    corr = df.corr()
    corr['selected'] = corr['trip_duration'].apply(lambda x: abs(x) > 0.08)
    print(f'train corrolation matrix:\n{corr[['trip_duration', 'selected']]}')
    selected_columns = corr[corr['selected']].index.tolist()
    selected_columns.append('pickup_month')
    selected_columns.append('pickup_hour')
    df = df[selected_columns]
    val_df = val_df[selected_columns]
    print(f'Features selected: {selected_columns}')

    
    return df, val_df



def data_preprocess():
    df = pd.read_csv('data/Train.csv')
    val_df = pd.read_csv('data/Val.csv')
    
    df, val_df = data_encoding(df, val_df)
    df, val_df = outliers_treatment(df.copy(), val_df.copy())
    df, val_df = drop_columns(df, val_df)
    df, val_df = feature_engineering(df, val_df)

    # # save the processed data
    # df = df[['distance', 'pickup_month', 'pickup_day', 'pickup_hour', 'trip_duration']]
    # val_df = val_df[['distance', 'pickup_month', 'pickup_day', 'pickup_hour', 'trip_duration']]
    # df.to_csv('processed_train.csv', index=False)
    # val_df.to_csv('processed_val.csv', index=False)
    
    return df, val_df

# if __name__ == '__main__':
#     data_preprocess()


