import os
import glob
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import scipy as sc
from sklearn.model_selection import KFold
# import lightgbm as lgb
import optuna
import optuna.integration.lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


# Function to calculate the root mean squared percentage error
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

# Function to early stop with root mean squared percentage error
def feval_rmspe(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return 'RMSPE', rmspe(y_true, y_pred), False


# ==============================
# Load data
# ==============================
# Read train data
train = pd.read_csv('train.csv')

# Drop NaN
train = train.dropna()

# Drop outlier
train = train.loc[train['log_return1_realized_volatility'] < 0.02]

# Select Features
select_features_df = pd.read_csv('RemoveOutliers_100_good_features.csv')
select_features = select_features_df['Feature'].values.tolist()

params = {
      #'device': 'gpu',
      'objective': 'regression',  
      'boosting_type': 'gbdt',
      "verbosity": -1,
      'n_jobs': -1,
    }

    
# Split features and target
x = train.drop(['row_id', 'target', 'time_id'], axis = 1)
x = x[select_features]
y = train['target']


# Transform stock id to a numeric value
x['stock_id'] = x['stock_id'].astype(int)

# Create out of folds array
oof_predictions = np.zeros(x.shape[0])

# Create a KFold object
kfold = KFold(n_splits = 10, random_state = 66, shuffle = True)
# Iterate through each fold
for fold, (trn_ind, val_ind) in enumerate(kfold.split(x)):
    print(f'Training fold {fold + 1}')
    x_train, x_val = x.iloc[trn_ind], x.iloc[val_ind]
    y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]
    # Root mean squared percentage error weights
    train_weights = 1 / np.square(y_train)
    val_weights = 1 / np.square(y_val)
    train_dataset = lgb.Dataset(x_train, y_train, weight = train_weights, categorical_feature = ['stock_id'])
    val_dataset = lgb.Dataset(x_val, y_val, weight = val_weights, categorical_feature = ['stock_id'])
    model = lgb.train(params = params, 
                    train_set = train_dataset, 
                    valid_sets = [train_dataset, val_dataset], 
                    num_boost_round = 10000, 
                    early_stopping_rounds = 50, 
                    verbose_eval = 50,
                    feval = feval_rmspe)
    # Add predictions to the out of folds array
    oof_predictions[val_ind] = model.predict(x_val)
    print(model.params)
    
rmspe_score = rmspe(y, oof_predictions)
print(f'Our out of folds RMSPE is {rmspe_score}')