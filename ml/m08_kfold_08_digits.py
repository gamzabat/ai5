import numpy as np
import pandas as pd

from sklearn.datasets import load_digits

from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.svm import SVC

#1 data
x, y = load_digits(return_X_y = True)

print(x.shape, y.shape)

kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 333)

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
# accuracy : [0.98333333 0.98888889 0.98328691 0.99442897 0.98607242]
# average : 0.9872

# StratifiedKFold
# accuracy : [0.98611111 0.99444444 0.98885794 0.98885794 0.98050139] 
#  average : 0.9878