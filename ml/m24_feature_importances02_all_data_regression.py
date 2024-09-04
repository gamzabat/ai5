from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_diabetes

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

import time

random_state = 7777

#1 data
california_datasets = fetch_california_housing()

diabetes_datasets = load_diabetes()

datasets = [california_datasets, diabetes_datasets]

for dataset in datasets:
    x = dataset.data
    y = dataset.target

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        train_size = 0.8,
        random_state = random_state
    )

    model1 = DecisionTreeRegressor(random_state = random_state)
    model2 = RandomForestRegressor(random_state = random_state)
    model3 = GradientBoostingRegressor(random_state = random_state)
    model4 = XGBRegressor(random_state = random_state)

    models = [model1, model2, model3, model4]

    for model in models:
        model.fit(x_train, y_train)

        print("-------------------", model.__class__.__name__, "-------------------")
        print('acc :', model.score(x_test, y_test))
        print(model.feature_importances_)