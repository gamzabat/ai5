import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.datasets import load_breast_cancer, load_wine, load_digits

random_state = 7777


#1 data
cancer_datasets = load_breast_cancer()
wine_datasets = load_wine()
digits_datasets = load_digits()

datasets = [cancer_datasets, wine_datasets, digits_datasets]

for dataset in datasets:
    x = dataset.data
    y = dataset.target

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        train_size = 0.8,
        random_state = random_state,
        stratify = y
    )

    model1 = DecisionTreeClassifier(random_state = random_state)
    model2 = RandomForestClassifier(random_state = random_state)
    model3 = GradientBoostingClassifier(random_state = random_state)
    model4 = XGBClassifier(random_state = random_state)

    models = [model1, model2, model3, model4]

    for model in models:
        model.fit(x_train, y_train)

        print("-------------------", model.__class__.__name__, "-------------------")
        print('acc :', model.score(x_test, y_test))
        print(model.feature_importances_)