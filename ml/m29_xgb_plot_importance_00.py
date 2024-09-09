from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

import numpy as np

import matplotlib.pyplot as plt

from xgboost.plotting import plot_importance

random_state = 7777

datasets = load_iris()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)

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

# def plot_feature_importances_dataset(model):
#     n_features = datasets.data.shape[1]

#     plt.barh(np.arange(n_features), model.feature_importances_, align = 'center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel("Feature Importances")
#     plt.ylabel("Features")
#     plt.ylim(-1, n_features)
#     plt.title(model.__class__.__name__)

# plot_feature_importances_dataset(model)

# plt.show()

plot_importance(model)

plt.show()