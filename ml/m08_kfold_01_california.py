from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR
import numpy as np
import xgboost as xgb

#1 data
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

kfold = KFold(n_splits = 5, shuffle = True, random_state = 333)

# model
model = xgb.XGBRegressor()

# train
scores = cross_val_score(
    model,
    x,
    y,
    cv = kfold
)

print('accuracy :', scores, '\n average :', np.round(np.mean(scores), 4))

# accuracy : [0.82742129 0.84140555 0.82123327 0.84284717 0.8478747 ] 
#  average : 0.8362