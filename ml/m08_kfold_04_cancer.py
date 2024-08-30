import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR

from sklearn.datasets import load_breast_cancer

#1 data
datasets = load_breast_cancer()

print(datasets)

print(datasets.DESCR)
print(datasets.feature_names)

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

# accuracy : [0.73296274 0.63335344 0.69905667 0.79938928 0.79171418]