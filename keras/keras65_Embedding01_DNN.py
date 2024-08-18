import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from sklearn.model_selection import train_test_split

# data
docs = [
    '너무 재미있다', '참 최고예요', '참 잘만든 영화예요',
    '추천하고 싶은 영화입니다.', '한 번 더 보고 싶어요.', '글쎄',
    '별로예요', '생각보다 지루하다', '연기가 어색해요',
    '재미없어요', '너무 재미없다', '참 재밋네요',
    '준영이 바보', '반장 잘 생겼다', '태운이 또 구라친다'
]

labels = np.array([1, 1, 1, 1, 1,
                   0, 0, 0, 0, 0,
                   0, 1, 0, 1, 0])

token = Tokenizer()

token.fit_on_texts(docs)

print(token.word_index)

x = token.texts_to_sequences(docs)

print(x)

x = pad_sequences(x, padding = 'pre')

print(x)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    labels,
    train_size = 0.90,
    random_state = 7777
)

x_question = ['태운이 참 재미없다.']

x_question = token.texts_to_sequences(x_question)

x_question = pad_sequences(x_question, padding = 'pre', maxlen = 5)

print(x_question)

# model
model = Sequential()

model.add(Dense(32, input_shape = (5, ), activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(
    x,
    labels,
    validation_split = 0.1,
#    callbacks = [es, mcp],
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

#loss : [2.690616085487818e-08, 1.0]
#predict : [[0.]]