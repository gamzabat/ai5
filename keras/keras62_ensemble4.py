import numpy as np

from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, Concatenate, concatenate
from sklearn.metrics import r2_score

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1 data
x1_datesets = np.array([range(100), range(301, 401)]).T
                       # 삼성 종가  하이닉스 종가

y1 = np.array(range(3001, 3101)) # 한강의 온도
y2 = np.array(range(13001, 13101)) # 비트코인의 가격

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1_datesets,
    y1,
    y2,
    train_size = 0.9,
    random_state = 7777
)

#2 model
# input1 = Input(shape=(2, ))

# dense1 = Dense(10, activation = 'relu', name = 'bit1')(input1)
# dense2 = Dense(10, activation = 'relu', name = 'bit2')(dense1)
# dense3 = Dense(10, activation = 'relu', name = 'bit3')(dense2)
# dense4 = Dense(10, activation = 'relu', name = 'bit4')(dense3)
# output1 = Dense(10, activation = 'relu', name = 'bit5')(dense4)

# last_output = Dense(10, name = 'last')(output1)

# divide1 = Dense(10)(last_output)
# divide1 = Dense(1)(divide1)

# divide2 = Dense(10)(last_output)
# divide2 = Dense(1)(divide2)

# model = Model(inputs = [input1], outputs = [divide1, divide2])

# #3 compile, fit
# model.compile(loss = 'mse', optimizer = 'adam')

# ##################### mcp 세이브 파일명 만들기 시작 ###################
# import datetime

# date = datetime.datetime.now()
# date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

# PATH = './_save/keras62/ensemble4/'

# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
# filepath = ''.join([PATH, 'k62_', date, "_", filename])
# ##################### mcp 세이브 파일명 만들기 끝 ###################
# mcp = ModelCheckpoint(
#     monitor = 'val_loss',
#     mode = 'auto',
#     verbose = 1,
#     save_best_only = True,
#     filepath = filepath
# )

# es = EarlyStopping(
#     monitor = 'val_loss',
#     mode = 'min',
#     patience = 100,
#     restore_best_weights = True
# )

# hist = model.fit(
#     x1_train,
#     [y1_train, y2_train],
#     validation_split = 0.2,
#     callbacks = [mcp],
#     batch_size = 8,
#     epochs = 1000
# )

#4 predict
model = load_model("./_save/k62_04.hdf5")

x1_pre = np.array([range(3101, 3106), range(101, 106)]).T
                       # 삼성 종가  하이닉스 종가

eval = model.evaluate(x1_test, [y1_test, y2_test])

print("loss :", eval)

result = model.predict(x1_pre)

print("예측값 :", result)

# loss : [0.03637947142124176, 0.0035889088176190853, 0.03279056400060654]