import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

text1 = '나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'
text2 = '태운이는 선생을 괴롭힌다. 준영이는 못 생겼다. 사영이는 마구 마구 더 못 생겼다.'

token = Tokenizer()

token.fit_on_texts([text1, text2])

print(token.word_index)
print(token.word_counts)

x = token.texts_to_sequences([text1, text2])

x = np.concatenate(x)

print(x)

y1 = to_categorical(x)[:, 1:]

print(y1)

y2 = OneHotEncoder(sparse = False).fit_transform(np.array(x).reshape(-1, 1))

print(y2)

y3 = pd.get_dummies(np.array(x).reshape(-1,))

print(y3)