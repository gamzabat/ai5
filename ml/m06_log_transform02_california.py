from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout
from keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# data
datasets = fetch_california_housing()

df = pd.DataFrame(datasets.data, columns = datasets.feature_names)

df['target'] = datasets.target

print(df)

# df.boxplot() # Population is wrong
# plt.show()

print(df.info())
print(df.describe().T)

# df['Population'].plot.box() # Series 이므로

# df['Population'].hist(bins=50)

# df['target'].hist(bins = 50)

# plt.show()

x = df.drop(['target'], axis = 1).copy()
y = df['target']

######################## Population 로그변환 ############################
x['Population'] = np.log1p(x['Population']) # 지수변환 np.exp1m
######################## Population 로그변환 ############################
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    random_state = 4123
)
######################## Population 로그변환 ############################
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)
######################## Population 로그변환 ############################

# model
model = LinearRegression()

# train
model.fit(
    x_train,
    y_train
)

# predict
score = model.score(x_test, y_test)

print('score :', score)

y_predict = model.predict(x_test)

print("r2 :", r2_score(y_test, y_predict))

# before log transform
# score : 0.6608714790060335

# after log transform
# score : 0.6643919818426876