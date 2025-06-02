import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
import argparse
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
# Localy
from preprocessingData import preprocess_data



parcer = argparse.ArgumentParser(description='ArgumentParser')
parcer.add_argument('--alpha', type=float, default=1)
parcer.add_argument('--degree', type=int, default=3)
parcer.add_argument('--randomState', type=int, default=42)
args = parcer.parse_args()
alpha_ = args.alpha
degree_ = args.degree
randomState = args.randomState


# Load Data from csv files
df = pd.read_csv('data/HPrice_train.csv')
df = df.drop(columns=['Id'], axis=1)  # Drop Id column from training data
test = pd.read_csv('data/HPrice_test.csv')
target = pd.read_csv('data/HPrice_target.csv')


# Preprocessing Data
df, test = preprocess_data(df, test)


# Training
df_y = np.array(df["SalePrice"]).reshape(-1, 1)
df_x = np.array(df.drop(columns=["SalePrice"], axis=1)).reshape(1460, df.shape[1] - 1)
print(f"Train Dataset shape: {df_x.shape}")



# Train test split
X_train, X_val, y_train, y_val = train_test_split(df_x, df_y, test_size=0.30, random_state=randomState) 




model = Pipeline([
    ('poly', PolynomialFeatures(degree=degree_)),
    ('scaler', MinMaxScaler()),
    ('ridge', Ridge(fit_intercept=True, alpha=alpha_))
])


# Fit the model
model.fit(X_train, y_train)

print("Training Dataset")
# exit()
print(f'\tscore: {model.score(X_train, y_train)}')

# Validation
print("\n\nValidation Dataset")
print(f'\tscore: {model.score(X_val, y_val)}')



