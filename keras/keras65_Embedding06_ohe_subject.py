import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Flatten, Conv1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split

# data
docs = [
    '너무 재미있다', '참 최고예요', '참 잘만든 영화예요',
    '추천하고 싶은 영화입니다.', '한 번 더 보고 싶어요.', '글쎄',
    '별로예요', '생각보다 지루하다', '연기가 어색해요',
    '재미없어요', '너무 재미없다', '참 재밋네요',
    '준영이 바보', '반장 잘생겼다', '태운이 또 구라친다',
    '세상의 짐을 본인이 모두 짊어지고 가는 사람',
    '아침은 꼭꼭 챙겨먹고 저녁은 가볍게 먹어요',
    '잃은 만큼 얻는 것이 인생이다',
    '쓴소리 잘하는 사람',
    '진정한 자존심은 자신에게 진실한 거야'
]

labels = np.array([1, 1, 1, 1, 1,
                   0, 0, 0, 0, 0,
                   0, 1, 0, 1, 0,
                   0, 1, 1, 0, 1])

token = Tokenizer()

token.fit_on_texts(docs)

print(token.word_index)

x = token.texts_to_sequences(docs)

print(x)

x = pad_sequences(x, padding = 'pre', truncating = 'pre', maxlen = 5)

print(x)

x = to_categorical(np.array(x).reshape(-1, )).reshape(20, 5, 56)

print(x)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    labels,
    train_size = 0.90,
    random_state = 7777
)

x_question = ['진정한 자존심은 재미없어요']

x_question = token.texts_to_sequences(x_question)

print(x_question)

x_question = pad_sequences(x_question, padding = 'pre', maxlen = 5)

print(x_question)

x_question = to_categorical(x_question, num_classes = 56)

print(x_question)

# model
model = Sequential()

model.add(Conv1D(32, 2, input_shape = (5, 56), padding = 'same'))
model.add(Conv1D(32, 2, input_shape = (5, 32), padding = 'same'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 64,
    restore_best_weights = True
)

model.fit(
    x,
    labels,
    validation_split = 0.1,
    callbacks = [es],
    epochs = 500,
    batch_size = 2,
    verbose = 1
)

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_question)

print('loss :', loss)
print('predict :', np.round(y_pred))
print('predict :', y_pred)

# loss : [0.6772061586380005, 0.5]
# predict : [[0.]]
# predict : [[0.49882272]]