from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale = 1.0 / 255,
    horizontal_flip = True,        # 수평 뒤집기 = 증폭(완전 다른 데이터가 하나 더 생겼다)
    vertical_flip = True,         # 수직 뒤집기 = 증폭
    # width_shift_range = 0.1,      # 평행 이동   = 증폭
    # height_shift_range = 0.1,     # 평행 이동 수직
    rotation_range = 15,           # 정해진 각도만큼 이미지 회전
    # zoom_range = 1.2,             # 축소 또는 확대
    # shear_range = 0.7,            # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환.
    fill_mode = 'nearest',        # 원래 있던 가까운 놈으로 채운다.
)

augment_size = 100

print(x_train.shape) # (60000, 28, 28)
print(x_train[0].shape) # (28, 28)

# plt.imshow(x_train[0], cmap='gray')
# plt.show()

xy_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28 * 28), augment_size).reshape(-1, 28, 28, 1),
    np.zeros(augment_size),
    batch_size = augment_size,
    shuffle = False
).next()

print(type(xy_data))

# print(len(xy_data))
# print(xy_data[0].shape)

plt.figure(figsize = (7, 7))

for i in range(49):
    plt.subplot(7, 7, i + 1)
    plt.imshow(xy_data[0][i], cmap = 'gray')

plt.show()