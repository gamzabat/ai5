import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.svm import SVC

# data
x, y = load_iris(return_X_y = True)

print(x)

n_splits = 5

kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 333)

# model
model = SVC()

# train
scores = cross_val_score(
    model,
    x,
    y,
    cv = kfold
)

print('accuracy :', scores, '\n average :', np.round(np.mean(scores), 4))

# SVC
# accuracy : [1.         0.86666667 1.         0.96666667 0.96666667]
#  average : 0.96

# StratifiedKFold
# accuracy : [0.93333333 0.96666667 0.93333333 1.         1.        ]
#  average : 0.9667