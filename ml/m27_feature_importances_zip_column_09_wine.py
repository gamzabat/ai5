from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

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

pencentile = np.percentile(model.feature_importances_, 35)

rm_index = []

for index, importance in enumerate(model.feature_importances_):
    if importance <= pencentile:
        rm_index.append(index)

x_train_pca = x_train[:, rm_index].copy()
pca = PCA(n_components = 1)
x_train_pca = pca.fit_transform(x_train_pca)

x_train = np.delete(x_train, [rm_index], axis = 1)
x_train = np.concatenate([x_train, x_train_pca], axis = 1)

x_test_pca = x_test[:, rm_index].copy()
x_test_pca = pca.transform(x_test_pca)

x_test = np.delete(x_test, [rm_index], axis = 1)
x_test = np.concatenate([x_test, x_test_pca], axis = 1)

model = RandomForestClassifier(random_state = random_state)

model.fit(x_train, y_train)

print("-------------------", model.__class__.__name__, "-------------------")
print('acc :', model.score(x_test, y_test))
print(model.feature_importances_)

# before remove column
# acc : 1.0

# after remove column (below 35%)
# acc : 0.9666666666666667

# after pca (below 35%)
# acc : 0.9333333333333333