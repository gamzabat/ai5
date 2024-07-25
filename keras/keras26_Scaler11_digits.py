import numpy as np
import pandas as pd

from sklearn.datasets import load_digits

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import time

#1 data
x, y = load_digits(return_X_y = True)

y = pd.get_dummies(y)

print(x.shape, y.shape)

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

#2 model
model = Sequential()

#model.add(Dense(128, input_dim = 64, activation = 'relu'))
model.add(Dense(128, input_shape = (64, ), activation = 'relu'))

model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))

model.add(Dense(10, activation = 'softmax'))

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
    validation_split = 0.25,
    callbacks = [es],
    epochs = 1000,
    batch_size = 4,
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
# loss : [0.09055327624082565, 0.9694444537162781]
# acc : 0.9694444444444444

# after minMaxScaler
# loss : [0.158413365483284, 0.9583333134651184]
# acc : 0.9583333333333334

# after standardScaler
# loss : [0.24239280819892883, 0.9444444179534912]
# acc : 0.9388888888888889

# after minAbsScaler
# loss : [0.14551538228988647, 0.9694444537162781]
# acc : 0.9666666666666667

# after robustScaler
# loss : [0.30181559920310974, 0.9472222328186035]
# acc : 0.9472222222222222