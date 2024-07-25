import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_breast_cancer

import matplotlib.pyplot as plt

import time

#1 data
datasets = load_breast_cancer()

print(datasets)

print(datasets.DESCR)
print(datasets.feature_names)

# y의 값이 0, 1이다

x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (569, 30) (569,)

print(np.unique(y, return_counts = True))

print(pd.DataFrame(y).value_counts())

print(pd.Series(y).value_counts())

print(pd.value_counts(y))

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.7,
    random_state = 3333
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

print(x_train.shape, y_train.shape) # (398, 30) (398,)

print(x_test.shape, y_test.shape) # (171, 30) (171,)

model = Sequential()

model.add(Dense(32, input_dim = 30, activation = 'relu'))

model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))

#3 compile
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 50,
    restore_best_weights = True
)

hist = model.fit(
    x_train,
    y_train,
    validation_split = 0.3,
    callbacks = [es],
    epochs = 1000,
    batch_size = 32,
    verbose = 2
)

end_time = time.time()

loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print(y_pred[:30])

print(np.round(y_pred[:30]))

print("loss :", loss)
print("acc :", accuracy_score(y_test, np.round(y_pred)))

# before scaler
# loss : [0.12934625148773193, 0.9649122953414917]
# acc : 0.9649122807017544

# after minMaxScaler
# loss : [0.09741540998220444, 0.9707602262496948]
# acc : 0.9707602339181286

# after standardScaler
# loss : [0.08955945074558258, 0.9766082167625427]
# acc : 0.9766081871345029

# after minAbsScaler
# loss : [0.09429660439491272, 0.9707602262496948]
# acc : 0.9707602339181286

# after robustScaler
# loss : [0.14115223288536072, 0.9824561476707458]
# acc : 0.9824561403508771