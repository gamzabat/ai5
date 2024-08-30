from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR
import numpy as np
import pandas as pd

#1 data
PATH = "./_data/kaggle/bike-sharing-demand/" # 절대경로

train_csv = pd.read_csv(PATH + "train.csv", index_col = 0)
test_csv = pd.read_csv(PATH + "test.csv", index_col = 0)

sampleSubmission_csv = pd.read_csv(PATH + "sampleSubmission.csv", index_col = 0)
#---------------------------------------------------------------
x = train_csv.drop(['casual', 'registered', 'count'], axis = 1) # [] -> python에서 list 형식이다

y = train_csv['count']

kfold = KFold(n_splits = 5, shuffle = True, random_state = 333)

# model
model = SVR()

# compile
scores = cross_val_score(
    model,
    x,
    y,
    cv = kfold
)

print('accuracy :', scores, '\n average :', np.round(np.mean(scores), 4))

# accuracy : [0.20161519 0.18059705 0.21608856 0.21685107 0.17728516]