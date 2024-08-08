import numpy as np

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, BatchNormalization, Dropout, Flatten

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model = load_model('./_save/k54_split4.hdf5')

x_predict = np.array(range(96, 106)).reshape(1, 5, 2)

y_pred = model.predict(x_predict)

print("pred :", y_pred)

# pred : [[80.57921]]