from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

import numpy as np

import matplotlib.pyplot as plt

random_state = 7777

datasets = load_diabetes()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)

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

def plot_feature_importances_dataset(models):
    n_features = datasets.data.shape[1]

    sub = 221

    for i in range(0, 4):
        plt.subplot(sub + i)
        plt.barh(np.arange(n_features), models[i].feature_importances_, align = 'center')
        plt.yticks(np.arange(n_features), datasets.feature_names)
        plt.xlabel("Feature Importances")
        plt.ylabel("Features")
        plt.ylim(-1, n_features)
        plt.title(models[i].__class__.__name__)

plot_feature_importances_dataset(models)

plt.show()