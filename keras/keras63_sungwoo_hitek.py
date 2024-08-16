import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, Concatenate, Conv1D, Flatten, Conv2D, MaxPooling2D, Dropout
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import RootMeanSquaredError

PATH_CSV = "./_data/중간고사데이터/"

#1 data
#=================================================================
def split_dataset(dataset, size):
    result = []

    for i in range(len(dataset) - size + 1):
        if (i % 10000 == 0):
            print(i)

        subset = dataset[i : (i + size)]

        result.append(subset)

    return np.array(result)
#=================================================================

PREDICT_DAYS = 5
FEATURE = 9
# -------------------------------------------------------------------------------------------------
import os.path

if os.path.isfile(PATH_CSV + "NAVER 240816.csv"):
    x1_datasets = pd.read_csv(PATH_CSV + "NAVER 240816.csv", index_col = 0, thousands = ",") # 네이버
else:
    x1_datasets = pd.read_csv(PATH_CSV + "NAVER_240816.csv", index_col = 0, thousands = ",") # 네이버

x1_datasets = x1_datasets.drop(['전일비', '등락률', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis = 1)

x1_datasets = x1_datasets.drop(x1_datasets.columns[4], axis = 1)

x1_datasets = x1_datasets[x1_datasets.index >= '2020/10/15']

x1_datasets.index = pd.to_datetime(x1_datasets.index, format = '%Y/%m/%d')

x1_datasets['day'] = x1_datasets.index.day
x1_datasets['month'] = x1_datasets.index.month
x1_datasets['year'] = x1_datasets.index.year
x1_datasets['dow'] = x1_datasets.index.dayofweek

x1_datasets.iloc[0, 0] = 159200.0
x1_datasets.iloc[0, 1] = 159200.0
x1_datasets.iloc[0, 2] = 157000.0
x1_datasets.iloc[0, 3] = 157500.0
x1_datasets.iloc[0, 4] = 813296.0

x1_datasets = x1_datasets[::-1]

# -------------------------------------------------------------------------------------------------
if os.path.isfile(PATH_CSV + "NAVER 240816.csv"):
    x2_datasets = pd.read_csv(PATH_CSV + "하이브 240816.csv", index_col = 0, thousands = ",") # 하이브
else:
    x2_datasets = pd.read_csv(PATH_CSV + "하이브_240816.csv", index_col = 0, thousands = ",") # 하이브

x2_datasets = x2_datasets.drop(['전일비', '등락률', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis = 1)

x2_datasets = x2_datasets.drop(x2_datasets.columns[4], axis = 1)

x2_datasets.index = pd.to_datetime(x2_datasets.index, format = '%Y/%m/%d')

x2_datasets['day'] = x2_datasets.index.day
x2_datasets['month'] = x2_datasets.index.month
x2_datasets['year'] = x2_datasets.index.year
x2_datasets['dow'] = x2_datasets.index.dayofweek

x2_datasets.iloc[0, 0] = 164700.0
x2_datasets.iloc[0, 1] = 168600.0
x2_datasets.iloc[0, 2] = 163500.0
x2_datasets.iloc[0, 3] = 166400.0
x2_datasets.iloc[0, 4] = 188123.0

x2_datasets = x2_datasets[::-1]
# -------------------------------------------------------------------------------------------------
if os.path.isfile(PATH_CSV + "NAVER 240816.csv"):
    x3_datasets = pd.read_csv(PATH_CSV + "성우하이텍 240816.csv", index_col = 0, thousands = ",") # 성우하이텍
else:
    x3_datasets = pd.read_csv(PATH_CSV + "성우하이텍_240816.csv", index_col = 0, thousands = ",") # 성우하이텍

x3_datasets = x3_datasets.drop(['전일비', '등락률', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis = 1)

x3_datasets = x3_datasets.drop(x3_datasets.columns[4], axis = 1)

x3_datasets = x3_datasets[x3_datasets.index >= '2020/10/15']

x3_datasets.index = pd.to_datetime(x3_datasets.index, format = '%Y/%m/%d')

x3_datasets['day'] = x3_datasets.index.day
x3_datasets['month'] = x3_datasets.index.month
x3_datasets['year'] = x3_datasets.index.year
x3_datasets['dow'] = x3_datasets.index.dayofweek

x3_datasets.iloc[0, 0] = 7580.0
x3_datasets.iloc[0, 1] = 7630.0
x3_datasets.iloc[0, 2] = 7350.0
x3_datasets.iloc[0, 3] = 7420.0
x3_datasets.iloc[0, 4] = 833336.0

x3_datasets = x3_datasets[::-1]
# -------------------------------------------------------------------------------------------------
x4_datasets = pd.read_csv(PATH_CSV + "KOSPI.csv", index_col = 0, thousands = ",") # 코스피

x4_datasets = x4_datasets[x4_datasets.index >= '2020-10-15']

x4_datasets.index = pd.to_datetime(x4_datasets.index, format = '%Y-%m-%d')

x4_datasets['day'] = x4_datasets.index.day
x4_datasets['month'] = x4_datasets.index.month
x4_datasets['year'] = x4_datasets.index.year
x4_datasets['dow'] = x4_datasets.index.dayofweek
# -------------------------------------------------------------------------------------------------
x5_datasets = pd.read_csv(PATH_CSV + "KOSDAQ.csv", index_col = 0, thousands = ",") # 코스닥

x5_datasets = x5_datasets[x5_datasets.index >= '2020-10-15']

x5_datasets.index = pd.to_datetime(x5_datasets.index, format = '%Y-%m-%d')

x5_datasets['day'] = x5_datasets.index.day
x5_datasets['month'] = x5_datasets.index.month
x5_datasets['year'] = x5_datasets.index.year
x5_datasets['dow'] = x5_datasets.index.dayofweek
# -------------------------------------------------------------------------------------------------
x6_datasets = pd.read_csv(PATH_CSV + "DJI.csv", index_col = 0, thousands = ",") # 다우존스

x6_datasets = x6_datasets[x6_datasets.index >= '2020-10-15']

x6_datasets.index = pd.to_datetime(x6_datasets.index, format = '%Y-%m-%d')

x6_datasets['day'] = x6_datasets.index.day
x6_datasets['month'] = x6_datasets.index.month
x6_datasets['year'] = x6_datasets.index.year
x6_datasets['dow'] = x6_datasets.index.dayofweek
# -------------------------------------------------------------------------------------------------
x7_datasets = pd.read_csv(PATH_CSV + "NASDAQ.csv", index_col = 0, thousands = ",") # 나스닥

x7_datasets = x7_datasets[x7_datasets.index >= '2020-10-15']

x7_datasets.index = pd.to_datetime(x7_datasets.index, format = '%Y-%m-%d')

x7_datasets['day'] = x7_datasets.index.day
x7_datasets['month'] = x7_datasets.index.month
x7_datasets['year'] = x7_datasets.index.year
x7_datasets['dow'] = x7_datasets.index.dayofweek
# -------------------------------------------------------------------------------------------------
x8_datasets = pd.read_csv(PATH_CSV + "SP500.csv", index_col = 0, thousands = ",") # S&P500

x8_datasets = x8_datasets[x8_datasets.index >= '2020-10-15']

x8_datasets.index = pd.to_datetime(x8_datasets.index, format = '%Y-%m-%d')

x8_datasets['day'] = x8_datasets.index.day
x8_datasets['month'] = x8_datasets.index.month
x8_datasets['year'] = x8_datasets.index.year
x8_datasets['dow'] = x8_datasets.index.dayofweek
# -------------------------------------------------------------------------------------------------
intersection = x1_datasets.index.intersection(x6_datasets.index).sort_values()

x1_datasets = x1_datasets[x1_datasets.index.isin(intersection)]
x2_datasets = x2_datasets[x2_datasets.index.isin(intersection)]
x3_datasets = x3_datasets[x3_datasets.index.isin(intersection)]
x4_datasets = x4_datasets[x4_datasets.index.isin(intersection)]
x5_datasets = x5_datasets[x5_datasets.index.isin(intersection)]
x6_datasets = x6_datasets[x6_datasets.index.isin(intersection)]
x7_datasets = x7_datasets[x7_datasets.index.isin(intersection)]
x8_datasets = x8_datasets[x8_datasets.index.isin(intersection)]

# print(x1_datasets, x1_datasets.shape)
# print(x2_datasets, x2_datasets.shape)
# print(x3_datasets, x3_datasets.shape)
# print(x4_datasets, x4_datasets.shape)
# print(x5_datasets, x5_datasets.shape)
# print(x6_datasets, x6_datasets.shape)
# print(x7_datasets, x7_datasets.shape)
# print(x8_datasets, x8_datasets.shape)
# -------------------------------------------------------------------------------------------------
x1_datasets = x1_datasets.values.astype(np.float64)
x2_datasets = x2_datasets.values.astype(np.float64)
x3_datasets = x3_datasets.values.astype(np.float64)
x4_datasets = x4_datasets.values.astype(np.float64)
x5_datasets = x5_datasets.values.astype(np.float64)
x6_datasets = x6_datasets.values.astype(np.float64)
x7_datasets = x7_datasets.values.astype(np.float64)
x8_datasets = x8_datasets.values.astype(np.float64)
# -------------------------------------------------------------------------------------------------
y = np.array(x3_datasets[:, 3]) # 성우하이텍 종가
# -------------------------------------------------------------------------------------------------
x1_scaler = MinMaxScaler()

x1_datasets = x1_scaler.fit_transform(x1_datasets)
# -------------------------------------------------------------------------------------------------
x2_scaler = MinMaxScaler()

x2_datasets = x2_scaler.fit_transform(x2_datasets)
# -------------------------------------------------------------------------------------------------
x3_scaler = MinMaxScaler()

x3_datasets = x3_scaler.fit_transform(x3_datasets)
# -------------------------------------------------------------------------------------------------
x4_scaler = MinMaxScaler()

x4_datasets = x4_scaler.fit_transform(x4_datasets)
# -------------------------------------------------------------------------------------------------
x5_scaler = MinMaxScaler()

x5_datasets = x5_scaler.fit_transform(x5_datasets)
# -------------------------------------------------------------------------------------------------
x6_scaler = MinMaxScaler()

x6_datasets = x6_scaler.fit_transform(x6_datasets)
# -------------------------------------------------------------------------------------------------
x7_scaler = MinMaxScaler()

x7_datasets = x7_scaler.fit_transform(x7_datasets)
# -------------------------------------------------------------------------------------------------
x8_scaler = MinMaxScaler()

x8_datasets = x8_scaler.fit_transform(x8_datasets)
# -------------------------------------------------------------------------------------------------
x1_datasets = split_dataset(x1_datasets, PREDICT_DAYS)
x2_datasets = split_dataset(x2_datasets, PREDICT_DAYS)
x3_datasets = split_dataset(x3_datasets, PREDICT_DAYS)
x4_datasets = split_dataset(x4_datasets, PREDICT_DAYS)
x5_datasets = split_dataset(x5_datasets, PREDICT_DAYS)
x6_datasets = split_dataset(x6_datasets, PREDICT_DAYS)
x7_datasets = split_dataset(x7_datasets, PREDICT_DAYS)
x8_datasets = split_dataset(x8_datasets, PREDICT_DAYS)

# x1_datasets = x1_datasets[:-1]
# x2_datasets = x2_datasets[:-1]
# x3_datasets = x3_datasets[:-1]
# x4_datasets = x4_datasets[:-1]
# x5_datasets = x5_datasets[:-1]
# x6_datasets = x6_datasets[:-1]
# x7_datasets = x7_datasets[:-1]
# x8_datasets = x8_datasets[:-1]

y = split_dataset(y, PREDICT_DAYS)

y = y[1:]
# -------------------------------------------------------------------------------------------------
x1_pred = x1_datasets[-1:]
x2_pred = x2_datasets[-1:]
x3_pred = x3_datasets[-1:]
x4_pred = x4_datasets[-1:]
x5_pred = x5_datasets[-1:]
x6_pred = x6_datasets[-1:]
x7_pred = x7_datasets[-1:]
x8_pred = x8_datasets[-1:]

answer = np.array([7420,])
# -------------------------------------------------------------------------------------------------
# x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, x4_train, x4_test,\
#     x5_train, x5_test, x6_train, x6_test, x7_train, x7_test, x8_train, x8_test, y_train, y_test = train_test_split(
#     x1_datasets,
#     x2_datasets,
#     x3_datasets,
#     x4_datasets,
#     x5_datasets,
#     x6_datasets,
#     x7_datasets,
#     x8_datasets,
#     y,
#     train_size = 0.9,
#     random_state = 7777
# )

# x1_train = x1_train.reshape(-1, PREDICT_DAYS, FEATURE, 1)
# x2_train = x2_train.reshape(-1, PREDICT_DAYS, FEATURE, 1)
# x3_train = x3_train.reshape(-1, PREDICT_DAYS, FEATURE, 1)
# x4_train = x4_train.reshape(-1, PREDICT_DAYS, FEATURE, 1)
# x5_train = x5_train.reshape(-1, PREDICT_DAYS, FEATURE, 1)
# x6_train = x6_train.reshape(-1, PREDICT_DAYS, FEATURE, 1)
# x7_train = x7_train.reshape(-1, PREDICT_DAYS, FEATURE, 1)
# x8_train = x8_train.reshape(-1, PREDICT_DAYS, FEATURE, 1)

# #2-1 model
# input01 = Input(shape = (PREDICT_DAYS, FEATURE, 1))

# dense01 = Conv2D(10, 2, activation = 'relu', name = 'bit0101', padding = 'same')(input01)
# dense01 = MaxPooling2D()(dense01)
# dense01 = Dense(20, activation = 'relu', name = 'bit0102')(dense01)

# output01 = Dense(30, activation = 'relu', name = 'bit0103')(dense01)

# #2-2 model
# input02 = Input(shape = (PREDICT_DAYS, FEATURE, 1))

# dense02 = Conv2D(10, 2, activation = 'relu', name = 'bit0201', padding = 'same')(input02)
# dense02 = MaxPooling2D()(dense02)
# dense02 = Dense(20, activation = 'relu', name = 'bit0202')(dense02)

# output02 = Dense(30, activation = 'relu', name = 'bit0203')(dense02)

# #2-3 model
# input03 = Input(shape = (PREDICT_DAYS, FEATURE, 1))

# dense03 = Conv2D(10, 2, activation = 'relu', name = 'bit0301', padding = 'same')(input03)
# dense03 = MaxPooling2D()(dense03)
# dense03 = Dense(20, activation = 'relu', name = 'bit0302')(dense03)

# output03 = Dense(30, activation = 'relu', name = 'bit0303')(dense03)

# #2-4 model
# input04 = Input(shape = (PREDICT_DAYS, FEATURE, 1))

# dense04 = Conv2D(10, 2, activation = 'relu', name = 'bit0401', padding = 'same')(input04)
# dense04 = MaxPooling2D()(dense04)
# dense04 = Dense(20, activation = 'relu', name = 'bit0402')(dense04)

# output04 = Dense(30, activation = 'relu', name = 'bit0403')(dense04)

# #2-5 model
# input05 = Input(shape = (PREDICT_DAYS, FEATURE, 1))

# dense05 = Conv2D(10, 2, activation = 'relu', name = 'bit0501', padding = 'same')(input05)
# dense05 = MaxPooling2D()(dense05)
# dense05 = Dense(20, activation = 'relu', name = 'bit0502')(dense05)

# output05 = Dense(30, activation = 'relu', name = 'bit0503')(dense05)

# #2-6 model
# input06 = Input(shape = (PREDICT_DAYS, FEATURE, 1))

# dense06 = Conv2D(10, 2, activation = 'relu', name = 'bit0601', padding = 'same')(input06)
# dense06 = MaxPooling2D()(dense06)
# dense06 = Dense(20, activation = 'relu', name = 'bit0602')(dense06)

# output06 = Dense(30, activation = 'relu', name = 'bit0603')(dense06)

# #2-7 model
# input07 = Input(shape = (PREDICT_DAYS, FEATURE, 1))

# dense07 = Conv2D(10, 2, activation = 'relu', name = 'bit0701', padding = 'same')(input07)
# dense07 = MaxPooling2D()(dense07)
# dense07 = Dense(20, activation = 'relu', name = 'bit0702')(dense07)

# output07 = Dense(30, activation = 'relu', name = 'bit0703')(dense07)

# #2-8 model
# input08 = Input(shape = (PREDICT_DAYS, FEATURE, 1))

# dense08 = Conv2D(10, 2, activation = 'relu', name = 'bit0801', padding = 'same')(input08)
# dense08 = MaxPooling2D()(dense08)
# dense08 = Dense(20, activation = 'relu', name = 'bit0802')(dense08)

# output08 = Dense(30, activation = 'relu', name = 'bit0803')(dense08)
# # -------------------------------------------------------------------------------------------------
# merge01 = Concatenate(name = 'merge0101')([output01, output02, output03, output04, output05, output06, output07, output08])

# merge01 = Flatten(name = 'merge0102')(merge01)

# merge01 = Dense(10, activation = 'relu', name = 'merge0103')(merge01)
# merge01 = Dense(20, activation = 'relu', name = 'merge0104')(merge01)
# merge01 = Dense(30, activation = 'relu', name = 'merge0105')(merge01)

# last_output = Dense(1, name = 'last')(merge01)

# model = Model(inputs = [input01, input02, input03, input04, input05, input06, input07, input08], outputs = last_output)

# model.summary()

# #3 compile, fit
# model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy', RootMeanSquaredError(name='rmse')])

# ##################### mcp 세이브 파일명 만들기 시작 ###################
# import datetime

# date = datetime.datetime.now()
# date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

# PATH = './_save/keras63/'

# filename = '{epoch:04d}-{val_rmse:.4f}.hdf5'
# filepath = ''.join([PATH, 'k63_', date, "_", filename])
# ##################### mcp 세이브 파일명 만들기 끝 ###################
# mcp = ModelCheckpoint(
#     monitor = 'val_rmse',
#     mode = 'auto',
#     verbose = 1,
#     save_best_only = True,
#     filepath = filepath
# )

# es = EarlyStopping(
#     monitor = 'val_rmse',
#     mode = 'min',
#     patience = 30,
#     restore_best_weights = True
# )

# class OnEpochEndPred(tf.keras.callbacks.Callback):
#     def __init__(self):
#         super().__init__()

#     def on_epoch_end(self, epoch, logs={}):
#         print("예측값 :", model.predict([x1_pred, x2_pred, x3_pred, x4_pred, x5_pred, x6_pred, x7_pred, x8_pred]))

# hist = model.fit(
#     [x1_train, x2_train, x3_train, x4_train, x5_train, x6_train, x7_train, x8_train],
#     y_train,
# #    validation_split = 0.25,
#     validation_data = ([x1_pred, x2_pred, x3_pred, x4_pred, x5_pred, x6_pred, x7_pred, x8_pred], answer),
#     callbacks = [es, mcp, OnEpochEndPred()],
#     batch_size = 32,
#     epochs = 10000
# )

# #4 predict
# eval = model.evaluate([x1_test, x2_test, x3_test, x4_test, x5_test, x6_test, x7_test, x8_test], y_test)

# print("loss :", eval)

model = load_model("./_save/k63_성우하이텍.hdf5")

result = model.predict([x1_pred, x2_pred, x3_pred, x4_pred, x5_pred, x6_pred, x7_pred, x8_pred])

print("예측값 :", result, answer - result)

# 예측값 : [[7568.6675]] [[148.66748047]]