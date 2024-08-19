import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Flatten, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split

# data
docs = [
    '너무 재미있다', '참 최고예요', '참 잘만든 영화예요',
    '추천하고 싶은 영화입니다.', '한 번 더 보고 싶어요.', '글쎄',
    '별로예요', '생각보다 지루하다', '연기가 어색해요',
    '재미없어요', '너무 재미없다', '참 재밋네요',
    '준영이 바보', '반장 잘생겼다', '태운이 또 구라친다'
]

labels = np.array([1, 1, 1, 1, 1,
                   0, 0, 0, 0, 0,
                   0, 1, 0, 1, 0])

token = Tokenizer()

token.fit_on_texts(docs)

print(token.word_index)

x = token.texts_to_sequences(docs)

x = pad_sequences(x, padding = 'pre')

print(x.shape)

x_question = ['태운이 참 재미없다']

x_question = token.texts_to_sequences(x_question)
x_question = pad_sequences(x_question, padding = 'pre', maxlen = 5)

# model
model = Sequential()

################### embedding 1 ###################
#model.add(Embedding(input_dim = 31, output_dim = 100, input_length = 5))
#  embedding (Embedding)       (None, 5, 100)            3100
#  lstm (LSTM)                 (None, 10)                4440
#  dense (Dense)               (None, 10)                110
#  dense_1 (Dense)             (None, 1)                 11

################### embedding 2 ###################
# model.add(Embedding(input_dim = 31, output_dim = 100))
#  embedding (Embedding)       (None, None, 100)         3100
#  lstm (LSTM)                 (None, 10)                4440
#  dense (Dense)               (None, 10)                110
#  dense_1 (Dense)             (None, 1)                 11

################### embedding 3 ###################
# model.add(Embedding(input_dim = 30, output_dim = 100))
# input_dim = 30 - 디폴트
# input_dim = 20 - 단어사전의 갯수보다 작을때, 연산량 줄고, 단어사전에서 임의로 빼서 - 성능조금저하
# input_dim = 40 - 단어사전의 갯수보다 클때, 연산량 늘고, 임의의 랜덤 임베딩 생성 - 성능조금저하

################### embedding 4 ###################
model.add(Embedding(31, 100)) # OK
# model.add(Embedding(31, 100, 5)) # error
# model.add(Embedding(31, 100, input_length = 5)) # OK
# model.add(Embedding(31, 100, input_length = 1)) # 5의 약수만 가능

model.add(LSTM(10, return_sequences = True)) # (None, 10)
model.add(LSTM(10)) # (None, 10)
model.add(Dense(10))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(
    x,
    labels,
#    validation_split = 0.1,
    epochs = 100,
    batch_size = 32,
    verbose = 1
)

loss = model.evaluate(x, labels, verbose = 0)

y_pred = model.predict(x_question)

print('loss :', loss)
print('predict :', np.round(y_pred))
print('predict :', y_pred)