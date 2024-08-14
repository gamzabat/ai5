import numpy as np

from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, Concatenate, concatenate
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1 data
x1_datesets = np.array([range(100), range(301, 401)]).T
                       # 삼성 종가  하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).T
                       # 원유, 환율, 금시세
x3_datasets = np.array([range(100), range(301, 401, ),
                        range(77, 177), range(33, 133)]).T

y = np.array(range(3001, 3101)) # 예나 강의 온도

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(
    x1_datesets,
    x2_datasets,
    x3_datasets,
    y,
    train_size = 0.9,
    random_state = 7777
)

#2-1 model
input1 = Input(shape=(2, ))

dense1 = Dense(10, activation = 'relu', name = 'bit1')(input1)
dense2 = Dense(20, activation = 'relu', name = 'bit2')(dense1)
dense3 = Dense(30, activation = 'relu', name = 'bit3')(dense2)
dense4 = Dense(40, activation = 'relu', name = 'bit4')(dense3)
output1 = Dense(50, activation = 'relu', name = 'bit5')(dense4)

#2-2 model
input11 = Input(shape=(3, ))

dense11 = Dense(100, activation = 'relu', name = 'bit11')(input11)
dense21 = Dense(200, activation = 'relu', name = 'bit21')(dense11)
output11 = Dense(300, activation = 'relu', name = 'bit10')(dense21)

#2-3 model
input31 = Input(shape=(4, ))

dense31 = Dense(40, activation = 'relu', name = 'bit31')(input31)
dense32 = Dense(50, activation = 'relu', name = 'bit32')(dense31)
dense33 = Dense(60, activation = 'relu', name = 'bit33')(dense32)
dense34 = Dense(70, activation = 'relu', name = 'bit34')(dense33)
output31 = Dense(60, activation = 'relu', name = 'bit35')(dense34)

merge1 = Concatenate(name = 'merge1')([output1, output11, output31])

merge2 = Dense(7, name = 'merge2')(merge1)
merge3 = Dense(20, name = 'merge3')(merge2)

last_output = Dense(1, name = 'last')(merge3)

model = Model(inputs = [input1, input11, input31], outputs = last_output)

model.summary()

#3 compile, fit
model.compile(loss = 'mse', optimizer = 'adam')

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()
date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras62/ensemble2/'

filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([PATH, 'k62_', date, "_", filename])
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
    patience = 16,
    restore_best_weights = True
)

hist = model.fit(
    [x1_train, x2_train, x3_train],
    y_train,
    validation_split = 0.2,
    callbacks = [es, mcp],
    batch_size = 1,
    epochs = 500
)

#4 predict
# model = load_model("./_save/k62_02.hdf5")

x1_pre = np.array([range(3101, 3106), range(101, 106)]).T
                       # 삼성 종가  하이닉스 종가
x2_pre = np.array([range(3301, 3306), range(2501, 2506), range(4001, 4006)]).T
                       # 원유, 환율, 금시세
x3_pre = np.array([range(5001, 5006), range(6001, 6006), range(5501, 5506), range(6501, 6506)]).T
                       # 원유, 환율, 금시세

eval = model.evaluate([x1_test, x2_test, x3_test], [y_test, y_test])

print("loss :", eval)

result = model.predict([x1_pre, x2_pre, x3_pre])

print("예측값 :", result)