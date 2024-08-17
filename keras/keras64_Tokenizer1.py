import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

text = "나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다."
#text = "셀룰러에서 셀룰러 데이터가 켜져 있지만 ‘개인용 핫스팟 설정’이 옵션으로 표시되지 않는 경우, 현재 요금제에 개인용 핫스팟을 추가할 수 있는지 이동통신사에 문의하십시오"

token = Tokenizer()

token.fit_on_texts([text])

print(token.word_index)
# {'셀룰러에서': 1, '셀룰러': 2, '데이터가': 3, '켜져': 4, '있지만': 5, '‘개인용': 6, '핫스팟': 7, '설정’이': 8, '옵션으로': 9, '표시되지': 10, '않는': 11, '경우': 12,
# '현재': 13, '요금제에': 14, '개인용': 15, '핫스팟을': 16, '추가할': 17, '수': 18, '있는지': 19, '이동통신사에': 20, '문의하십시오': 21}

print(token.word_counts)
# OrderedDict([('셀룰러에서', 1), ('셀룰러', 1), ('데이터가', 1), ('켜져', 1), ('있지만', 1), ('‘개인용', 1), ('핫스팟', 1), ('설정’이', 1), ('옵션으로', 1), ('표시되지', 1),
# ('않는', 1), ('경우', 1), ('현재', 1), ('요금제에', 1), ('개인용', 1), ('핫스팟을', 1), ('추가할', 1), ('수', 1), ('있는지', 1), ('이동통신사에', 1), ('문의하십시오', 1)])

x = token.texts_to_sequences([text])

print(x) # [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]

y1 = to_categorical(x)[:, :, 1:]

print(y1)

y2 = OneHotEncoder(sparse = False).fit_transform(np.array(x).reshape(-1, 1))

print(y2)

y3 = pd.get_dummies(np.array(x).reshape(-1,))

print(y3)