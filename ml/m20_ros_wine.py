import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import tensorflow as tf

CFG = {
    'SEED' : 7777
}

def set_seed(seed=14):
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(CFG['SEED'])

# data
datasets = load_wine()

x = datasets.data
y = datasets['target']

print(x.shape, y.shape) # (178, 13) (178,)
print(np.unique(y, return_counts = True)) # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
print(pd.value_counts(y))

print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

x = x[:-39]
y = y[:-39]

print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2]

print(np.unique(y, return_counts = True)) # (array([0, 1, 2]), array([59, 71,  8], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = .75,
    shuffle = True,
    random_state = 333,
    stratify = y
)

'''
# model
model = Sequential()

model.add(Dense(32, input_shape = (13, )))
model.add(Dense(3, activation = 'softmax'))

# train
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])

model.fit(
    x_train,
    y_train,
    epochs = 100,
    validation_split = 0.2
)

# predict
results = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test).argmax(axis = 1)

acc = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict, average = 'macro')

print('acc :', acc)
print('f1 :', f1)

# acc : 0.8285714285714286
# f1 : 0.7847222222222222
'''

################### SMOTE 적용 ###################
# pip install imblearn
from imblearn.over_sampling import RandomOverSampler

import sklearn as sk

print('scikit version :', sk.__version__)

print('before smote', np.unique(y_train, return_counts = True))

ros = RandomOverSampler(random_state = 7777)

# smote = SMOTE(random_state = 7777)

# x_train, y_train = smote.fit_resample(x_train, y_train)
x_train, y_train = ros.fit_resample(x_train, y_train)

print('after smote', np.unique(y_train, return_counts = True))

print(pd.value_counts(y_train))
################### SMOTE END ###################

# model
model = Sequential()

model.add(Dense(32, input_shape = (13, )))
model.add(Dense(3, activation = 'softmax'))

# train
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])

model.fit(
    x_train,
    y_train,
    epochs = 100,
    validation_split = 0.2
)

# predict
results = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test).argmax(axis = 1)

acc = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict, average = 'macro')

print('acc :', acc)
print('f1 :', f1)