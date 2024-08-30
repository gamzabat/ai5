import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR

PATH = "./_data/dacon/diabetes/" # 상대경로

#1 data
train_csv = pd.read_csv(PATH + "train.csv", index_col = 0)

train_csv = train_csv[train_csv['BloodPressure'] != 0]
train_csv = train_csv[train_csv['BMI'] > 0.0]
train_csv = train_csv[train_csv['Glucose'] > 0]
 
x = train_csv.drop(['Outcome'], axis = 1)

y = train_csv['Outcome']

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

# accuracy : [0.17604656 0.04838582 0.14502799 0.29444185 0.37798828]