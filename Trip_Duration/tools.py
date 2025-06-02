import numpy as np

# outlier treatment function
def wescr(column):
    q1, q3 = np.percentile(column, [25,75])
    iqr = q3 - q1
    lw = q1 - 1.5*iqr
    uw = q3 + 1.5*iqr
    return lw, uw


# lw, uw =wescr(df['trip_duration'])
# print(f'lw: {lw} | uw: {uw}')
# df['trip_duration'] = np.where(df['trip_duration']<lw,lw,df['trip_duration'])
# df['trip_duration'] = np.where(df['trip_duration']>uw,uw,df['trip_duration'])