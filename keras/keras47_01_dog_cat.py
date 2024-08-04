from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt

import numpy as np

path = './_data/image/me/me80.npy'

me = np.load(path)

print(me.shape)

model = load_model('./_save/k42_dog_cat.hdf5')

y_submit = model.predict(me)

print(y_submit)
print(np.round(y_submit))