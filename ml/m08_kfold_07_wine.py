import numpy as np
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.svm import SVC

#1 data
dataset = load_wine()

x = dataset.data
y = dataset.target

kfold = KFold(n_splits = 5, shuffle = True, random_state = 333)

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
# accuracy : [0.69444444 0.63888889 0.61111111 0.74285714 0.62857143] 
#  average : 0.6632

# StratifiedKFold
# accuracy : [0.63888889 0.69444444 0.72222222 0.62857143 0.65714286] 
#  average : 0.6683