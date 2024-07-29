import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_breast_cancer

#1 data
datasets = load_breast_cancer()

print(datasets)

print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    random_state = 3333
)

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train)

x_train = min_max_scaler.transform(x_train)
x_test = min_max_scaler.transform(x_test)

# model = Sequential()

# model.add(Dense(32, input_dim = 30, activation = 'relu'))

# model.add(Dense(16, activation = 'relu'))
# model.add(Dense(16, activation = 'relu'))
# model.add(Dense(16, activation = 'relu'))
# model.add(Dense(16, activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation = 'relu'))
# model.add(Dense(16, activation = 'relu'))
# model.add(Dense(16, activation = 'relu'))
# model.add(Dense(16, activation = 'relu'))
# model.add(Dense(16, activation = 'relu'))
# model.add(Dense(16, activation = 'relu'))

# model.add(Dense(1, activation = 'sigmoid'))

input1 = Input(shape = (30,))

dense1 = Dense(32, activation = 'relu')(input1)
dense2 = Dense(16, activation = 'relu')(dense1)
dense3 = Dense(16, activation = 'relu')(dense2)
dense4 = Dense(16, activation = 'relu')(dense3)
dense5 = Dense(16, activation = 'relu')(dense4)
drop1 = Dropout(0.2)(dense5)
dense6 = Dense(16, activation = 'relu')(drop1)
dense7 = Dense(16, activation = 'relu')(dense6)
dense8 = Dense(16, activation = 'relu')(dense7)
dense9 = Dense(16, activation = 'relu')(dense8)
dense10 = Dense(16, activation = 'relu')(dense9)
dense11 = Dense(16, activation = 'relu')(dense10)

output1 = Dense(1, activation = 'sigmoid')(dense11)

model = Model(inputs = input1, outputs = output1)

#3 compile
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

print(date) # 2024-07-26 16:49:36.336699
print(type(date)) # <class 'datetime.datetime'>

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras32/06-cancer/'

filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

filepath = ''.join([PATH, 'k32_', date, "_", filename])
##################### mcp 세이브 파일명 만들기 끝 ###################

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose = 1,
    save_best_only = True,
    filepath = filepath
)

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 64,
    restore_best_weights = True
)

hist = model.fit(
    x_train,
    y_train,
    validation_split = 0.3,
    callbacks = [es, mcp],
    epochs = 1000,
    batch_size = 32,
    verbose = 2
)

loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print("loss :", loss)
print("acc :", accuracy_score(y_test, np.round(y_pred)))

# before dropout
# loss : [0.11287593841552734, 0.9649122953414917]
# acc : 0.9649122807017544

# after dropout
# loss : [0.10897455364465714, 0.9736841917037964]
# acc : 0.9736842105263158