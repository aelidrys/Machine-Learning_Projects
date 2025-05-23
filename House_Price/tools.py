import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def wescr(column):
    q1, q3 = np.percentile(column, [25,75])
    iqr = q3 - q1
    lw = q1 - 1.5*iqr
    uw = q3 + 1.5*iqr
    return lw, uw


def outleir_treatment(df: pd.DataFrame):
    # int features outlier treatment
    columns = df.select_dtypes(include="int64").columns
    for i in columns:
        lw, uw = wescr(df[i])
        if lw == uw:
            continue
        df[i] = np.where(df[i]<lw,lw,df[i])
        df[i] = np.where(df[i]>uw,uw,df[i])

    # float features outlier treatment
    clmns2 = ['LotFrontage', 'GarageYrBlt', 'MasVnrArea']
    for i in clmns2:
        lw, uw = wescr(df[i])
        df[i] = np.where(df[i]<lw, lw, df[i])
        df[i] = np.where(df[i]>uw, uw, df[i])
        
        
        
def encoding_data(df, test):
    label_encoder = LabelEncoder()
    for i in df.select_dtypes(include="object").columns:
        df[i] = label_encoder.fit_transform(df[i])

    label_encoder = LabelEncoder()
    for i in test.select_dtypes(include="object").columns:
        test[i] = label_encoder.fit_transform(test[i])