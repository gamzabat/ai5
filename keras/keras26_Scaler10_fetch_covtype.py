import numpy as np
import pandas as pd

from sklearn.datasets import fetch_covtype

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import time

from sklearn.preprocessing import OneHotEncoder

#1 data
dataset = fetch_covtype()

x = pd.DataFrame(dataset.data).iloc[:, :13]

y = pd.get_dummies(dataset.target)

#y = pd.DataFrame(to_categorical(dataset.target))

print(y.head())

# y = OneHotEncoder(sparse = False).fit_transform(dataset.target.reshape(-1, 1))

# print(x)
# print(y.shape)

# print(pd.DataFrame(y).value_counts())

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    random_state = 7777,
    stratify = y
)

# min_max_scaler = MinMaxScaler()

# min_max_scaler.fit(x_train)

# x_train = min_max_scaler.transform(x_train)
# x_test = min_max_scaler.transform(x_test)

# standard_scaler = StandardScaler().fit(x_train)

# x_train = standard_scaler.transform(x_train)
# x_test = standard_scaler.transform(x_test)

# max_abs_scaler = MaxAbsScaler().fit(x_train)

# x_train = max_abs_scaler.transform(x_train)
# x_test = max_abs_scaler.transform(x_test)

robust_scaler = RobustScaler().fit(x_train)

x_train = robust_scaler.transform(x_train)
x_test = robust_scaler.transform(x_test)

# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

# print(pd.DataFrame(y_train).value_counts())

#2 model
model = Sequential()

model.add(Dense(64, input_dim = 13, activation = 'relu'))

model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))

model.add(Dense(7, activation = 'softmax'))

#3 compile
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 8,
    restore_best_weights = True
)

model.fit(
    x_train,
    y_train,
    validation_split = 0.25,
    callbacks = [es],
    epochs = 512,
    batch_size = 1024,
    verbose = 1
)

end_time = time.time()

print("fit time :", round(end_time - start_time, 2), "초")

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print(np.round(y_pred[:10]))

print("loss :", loss)
print("acc :", accuracy_score(y_test, np.round(y_pred)))

# before scaler
# loss : [0.33425068855285645, 0.8608469367027283]
# acc : 0.8554684474583273

# after minMaxScaler
# loss : [0.26443609595298767, 0.8916465044021606]
# acc : 0.8876621085514144

# after standardScaler
# loss : [0.20973464846611023, 0.9155185222625732]
# acc : 0.9134187585518446

# after minAbsScaler
# loss : [0.24365977942943573, 0.9005619287490845]
# acc : 0.8973262308202026

# after robustScaler
# loss : [0.21320636570453644, 0.9144514203071594]
# acc : 0.9119730127449377