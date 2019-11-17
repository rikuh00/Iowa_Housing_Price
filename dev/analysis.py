#%% IMPORTS
import pandas as pd
import os
import datetime as dt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

#%% IMPORT DATA
parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(parent)
data_path = os.path.join(parent, 'data','train.csv')
df_house = pd.read_csv(data_path)

#%% Data Summary
print(df_house.describe())

avg_lot_size = df_house.LotArea.mean()
print('The average lot size is {:.0f}'.format(avg_lot_size))

newest_house_year = df_house.YearBuilt.max()
current_year = dt.datetime.now().year
newest_house_age = current_year - newest_house_year
print('The newest house is {:.0f} years old'.format(newest_house_age))
print('\n')

#%% MODELLING
# FUNCTION - Print Functions for Outputs
def print_price_prediction(X, y, house_price_prediction):
    print('Sale Price Prediction for the first 5 houses:')
    print(X.head(5).reset_index(drop = True))
    print('\n')
    for i in range(0,5):
        print('{0}: ${1:.2f}'.format(i, house_price_prediction[i]))
    print('Mean absolute error: ${:.2f}'.format(mean_absolute_error(y, house_price_prediction)))
    print('\n')
    return None

# Define Target & Features
y = df_house.SalePrice
feature_list = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X = df_house[feature_list]

# Establish a Model (In-Sample)
house_price_model = DecisionTreeRegressor(random_state = 0)
house_price_model.fit(X, y)
house_price_prediction = house_price_model.predict(X)
print_price_prediction(X, y, house_price_prediction) #model validation

# Establish a Model (Out-of-Sample)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
house_price_model.fit(train_X, train_y)
house_price_prediction = house_price_model.predict(val_X)
print_price_prediction(train_X, val_y, house_price_prediction) #model validation

#%% EXPERIMENT - MAE VS MAX # OF LEAF NODES

# FUNCTION - Get MAE Values
def _get_mae_vs_node (max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    model.fit(train_X, train_y)
    model_prediction = model.predict(val_X)
    mae = mean_absolute_error(val_y, model_prediction)
    return mae

def get_mae_dict (train_X, val_X, train_y, val_y):
    max_leaf_nodes = list(range(10, 1000, 10))
    mae_dict = {max_leaf_node: _get_mae_vs_node(max_leaf_node, train_X, val_X, train_y, val_y) for max_leaf_node in max_leaf_nodes}
    return mae_dict

# FUNCTION - Get Minimum MAE and Corresponding Max. # of Nodes
def get_min_mae (mae_dict):
    min_mae_node = min(mae_dict, key = mae_dict.get)
    min_mae = mae_dict[min_mae_node]
    print('Min MAE = ${0:.2f} at max. {1} nodes'.format(min_mae, min_mae_node))
    return min_mae_node, min_mae

# FUNCTION - Plot MAE vs # of Nodes
def plot_mae_vs_node(mae_dict):
    df_mae = (pd.DataFrame(mae_dict, index = [0])).transpose()
    df_mae.rename(columns = {0:'MAE'}, inplace = True)
    plt.close()
    df_mae.plot()
    plt.title('MAE vs. Max. Number of Leaf Nodes')
    plt.ylabel('MAE ($)')
    plt.xlabel('Max. # of Leaf Nodes')
    plt.tight_layout()
    plt.savefig(r'output\MAE_vs_numNodes.png')
    plt.show()
    return None

# Summarize MAE vs. Max # of Nodes Info
mae_dict = get_mae_dict(train_X, val_X, train_y, val_y)
min_mae_node, min_mae = get_min_mae(mae_dict)
plot_mae_vs_node(mae_dict)

# Optimize House Pricing Model
house_price_model = DecisionTreeRegressor(max_leaf_nodes = min_mae_node, random_state = 0)
house_price_model.fit(train_X, train_y)
house_price_prediction = house_price_model.predict(val_X)
print_price_prediction(val_X, val_y, house_price_prediction)
