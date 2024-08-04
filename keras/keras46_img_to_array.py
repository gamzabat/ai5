from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img # 이미지 가져오기
from tensorflow.keras.preprocessing.image import img_to_array # 이미지 수치화

from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt

import numpy as np

path = './_data/image/me/cc.png'

img = load_img(path, target_size = (80, 80))

print(img)
print(type(img))

# plt.imshow(img)
# plt.show()

arr = img_to_array(img)

print(arr)
print(arr.shape) # (100, 100, 3)

#차원 확장
img = np.expand_dims(arr, axis = 0)

print(img.shape)

np.save('./_data/image/me/cc80.npy', img)

# me 폴더에 img를 npy로 저장할 것