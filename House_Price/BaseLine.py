import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
import argparse
from sklearn.metrics import mean_squared_error, r2_score

# Localy
from tools import outleir_treatment, encoding_data



parcer = argparse.ArgumentParser(description='ArgumentParser')
parcer.add_argument('--alpha', type=float, default=1)
parcer.add_argument('--degree', type=int, default=3)
args = parcer.parse_args()
alpha_ = args.alpha
degree_ = args.degree



# Load Data from csv files
df = pd.read_csv('HPrice_train.csv')
test = pd.read_csv('HPrice_test.csv')
test_y = pd.read_csv('HPrice_target.csv')



## Missing Values Treatment
# Drop the columns that have more than 30% of values missing 
df = df[[col for col in df.columns if df[col].isnull().sum() < 0.3 * df.shape[0]]]
test = test[[col for col in test.columns if test[col].isnull().sum() < 0.3 * test.shape[0]]]

# Use Mean or Medin or Mode to fill the remaining missing values or None Available
df['LotFrontage'].fillna(df['LotFrontage'].mean(), inplace=True)
df['MasVnrArea'].fillna(df['MasVnrArea'].mean(), inplace=True)
df['Electrical'].fillna("SBrkr", inplace=True) # fill by Mode
df['GarageYrBlt'].fillna(2005.0, inplace=True) # fill by Mode
df.fillna({'GarageType': 'NA', 'GarageFinish': 'NA', 'GarageQual': 'NA', 'GarageCond': 'NA'},
          inplace=True)
df.fillna({'BsmtQual':'NA', 'BsmtCond': 'NA', 'BsmtExposure': 'NA', 'BsmtFinType1': 'NA',
           'BsmtFinType2': 'NA'}, inplace=True)
# Missing Values in Test
test['LotFrontage'].fillna(test['LotFrontage'].mean(), inplace=True)
test['MasVnrArea'].fillna(test['MasVnrArea'].mean(), inplace=True)
test.fillna({'GarageType': 'NA', 'GarageFinish': 'NA', 'GarageQual': 'NA', 'GarageCond': 'NA'},
          inplace=True)
test.fillna({'BsmtQual':'NA', 'BsmtCond': 'NA', 'BsmtExposure': 'NA', 'BsmtFinType1': 'NA',
           'BsmtFinType2': 'NA'}, inplace=True)
test.fillna(test.mode().iloc[0], inplace=True)

# Outleirs Treatment
outleir_treatment(df)
outleir_treatment(test)

# Data Encoding
encoding_data(df, test)




# Slect Featuresc
corr = df.corr()
selected_columns = corr[(corr.iloc[-1]>0.20) | (corr.iloc[-1]<-0.20)].index
selected_columns.shape
df = df[selected_columns]
test_id = test[['Id']]
test = test[df.drop(columns='SalePrice').columns]


# Training
df_y = df["SalePrice"]
df_x = df.drop(columns=["SalePrice"], axis=1)

df_x = PolynomialFeatures(degree=degree_).fit_transform(df_x)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.30, random_state=15) 
print("X_train.shape : ", X_train.shape)
print("y_train.shape: ", y_train.shape)
print("X_test.shape: ", X_test.shape)
print("y_test.shape: ", y_test.shape)

# Scaling
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)

# fit LinearRegression model
linearReg = Ridge(fit_intercept=True, alpha=alpha_).fit(X_train, y_train)

print("train Dataset")
print(f'\tscore: {linearReg.score(X_train, y_train)}')


# Testing
X_test = scaler.transform(X_test)
y_predict = linearReg.predict(X_test)


print("\n\ntest Dataset")
print(f'\tscore: {linearReg.score(X_test, y_test)}')


test_x = np.array(test)
test_x = PolynomialFeatures(degree=degree_).fit_transform(test_x)
test_x = scaler.transform(test_x)

predict = linearReg.predict(np.array(test_x))



test_target = np.array(test_y['SalePrice'])

r2 = r2_score(test_target, predict)
print(f'Final Test')
# print(f'\tpr shape: {test_target.shape}')
# print(f'\ttrgt shape: {predict.shape}')
print(f'\tscore: {r2}')
