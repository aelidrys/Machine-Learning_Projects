import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
import argparse
from sklearn.metrics import mean_squared_error, r2_score

# Localy
from preprocessingData import preprocess_data



parcer = argparse.ArgumentParser(description='ArgumentParser')
parcer.add_argument('--alpha', type=float, default=1)
parcer.add_argument('--degree', type=int, default=3)
args = parcer.parse_args()
alpha_ = args.alpha
degree_ = args.degree


# Load Data from csv files
df = pd.read_csv('data/HPrice_train.csv')
test = pd.read_csv('data/HPrice_test.csv')
test_y = pd.read_csv('data/HPrice_target.csv')


# Preprocessing Data
df, test, test_id = preprocess_data(df, test)


# Training
df_y = df["SalePrice"]
df_x = df.drop(columns=["SalePrice"], axis=1)

df_x = PolynomialFeatures(degree=degree_).fit_transform(df_x)

# Train test split
X_train, X_val, y_train, y_test = train_test_split(df_x, df_y, test_size=0.30, random_state=15) 
print("X_train.shape : ", X_train.shape)
print("y_train.shape: ", y_train.shape)
print("X_val.shape: ", X_val.shape)
print("y_test.shape: ", y_test.shape)

# Scaling
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)

# fit LinearRegression model
linearReg = Ridge(fit_intercept=True, alpha=alpha_).fit(X_train, y_train)

print("train Dataset")
print(f'\tscore: {linearReg.score(X_train, y_train)}')


# Testing
X_val = scaler.transform(X_val)
y_predict = linearReg.predict(X_val)


print("\n\ntest Dataset")
print(f'\tscore: {linearReg.score(X_val, y_test)}')


test_x = np.array(test)
test_x = PolynomialFeatures(degree=degree_).fit_transform(test_x)
test_x = scaler.transform(test_x)

predict = linearReg.predict(np.array(test_x))



test_target = np.array(test_y['SalePrice'])

r2 = r2_score(test_target, predict)
print(f'Final Test')
print(f'\tscore: {r2}')
