#%% IMPORTS
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import os
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
#%% SETTING UP DATA

# FUNCTION - Read Dataset
def read_data(dataset_name):
    data_path = os.path.join(parent, 'data', dataset_name)
    df = pd.read_csv(data_path, index_col = 'Id')
    return df

# Reading Data
parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(parent)
df_train = read_data('train.csv')
df_test = read_data('test.csv')

# Targets and Predictors
y = df_train.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = df_train[features].copy()
X_test = df_test[features].copy()

# Creating Validation Sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 0)

#%% MODEL EVALUATION

# FUNCTION - get mean absolute error
def get_mae(model, model_num):
    prediction = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, prediction)
    print('MAE for Model # {0}: ${:.2f}'.format(model_num, mae))
    return mae

model_1 = RandomForestRegressor()