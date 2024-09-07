import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score

import warnings

warnings.filterwarnings('ignore')

# data
x, y = load_iris(return_X_y = True)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    random_state = 3377,
    train_size = 0.8
    stratify =y
)


scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# model
model = XGBClassifier(
    n_estimators = 500,
    max_depth = 6,
    gamma = 0,
    min_child_weight = 0,
    subsample = 0.4,
    reg_alpah = 0, # L1 규제
    reg_lambda = 1 # L2 규제
)

model.fit(
    x_train,
    y_train,
    eval_sets = ([x_test, y_test]),
    eval_metrics = "mlogloss",
    verbose = 1
)

model.score(
    x_test,
    y_test
)