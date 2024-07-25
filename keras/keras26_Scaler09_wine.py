import numpy as np
import pandas as pd

from sklearn.datasets import load_wine

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import time

#1 data
dataset = load_wine()

x = dataset.data
y = pd.get_dummies(dataset.target)

print(x)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    random_state = 5555
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

model.add(Dense(3, activation = 'softmax'))

#3 compile
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 64,
    restore_best_weights = True
)

model.fit(
    x_train,
    y_train,
    validation_split = 0.3,
    callbacks = [es],
    epochs = 1000,
    batch_size = 4,
    verbose = 2
)

end_time = time.time()

loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print(np.round(y_pred[:10]))

print("loss :", loss)
print("acc :", accuracy_score(y_test, np.round(y_pred)))

# before scaler
# loss : [0.7308107614517212, 0.6944444179534912]
# acc : 0.5

# after minMaxScaler
# loss : [0.2496759295463562, 0.9166666865348816]
# acc : 0.9166666666666666

# after standardScaler
# loss : [0.2565852701663971, 0.9444444179534912]
# acc : 0.9444444444444444

# after minAbsScaler
# loss : [0.3894861936569214, 0.9166666865348816]
# acc : 0.9166666666666666

# after robustScaler
# loss : [0.35075727105140686, 0.9722222089767456]
# acc : 0.9722222222222222