from tensorflow.keras.models import load_model

import numpy as np

path = './_data/image/me/cc100.npy'

me = np.load(path)

print(me.shape)

model = load_model('./_save/k49_faces2.hdf5')

y_submit = model.predict(me)

print(model.predict(me))
print(np.round(y_submit))