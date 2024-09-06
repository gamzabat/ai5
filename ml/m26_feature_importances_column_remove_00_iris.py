from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

import numpy as np

random_state = 7777

x, y = load_iris(return_X_y = True)

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    random_state = random_state,
    stratify = y
)

model = RandomForestClassifier(random_state = random_state)

model.fit(x_train, y_train)

pencentile = np.percentile(model.feature_importances_, 5)

rm_index = []

for index, importance in enumerate(model.feature_importances_):
    if importance <= pencentile:
        rm_index.append(index)

x_train = np.delete(x_train, [rm_index], axis = 1)
x_test = np.delete(x_test, [rm_index], axis = 1)

model = RandomForestClassifier(random_state = random_state)

model.fit(x_train, y_train)

print("-------------------", model.__class__.__name__, "-------------------")
print('acc :', model.score(x_test, y_test))
print(model.feature_importances_)

# ------------------- RandomForestClassifier -------------------
# acc : 1.0
# [0.09522768 0.03243642 0.43290218 0.43943372]