from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR
import numpy as np

#1 data
datasets = load_diabetes()

x = datasets.data
y = datasets.target

kfold = KFold(n_splits = 5, shuffle = True, random_state = 333)

# model
model = SVR()

# train
scores = cross_val_score(
    model,
    x,
    y,
    cv = kfold
)

print('accuracy :', scores, '\n average :', np.round(np.mean(scores), 4))

# accuracy : [0.16365784 0.16866935 0.17705885 0.07020419 0.19004339]