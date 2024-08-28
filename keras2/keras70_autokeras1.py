import autokeras as ak
import tensorflow as tf
import time

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print(ak.__version__) # 1.0.20
print(tf.__version__) # 2.7.4

# data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(x_train.shape, x_test.shape)

# model
model = ak.ImageClassifier(
    overwrite = False,
    max_trials = 3
)

# train
start_time = time.time()

model.fit(
    x_train,
    y_train,
    epochs = 10,
    validation_split = 0.15
)

end_time = time.time()

################################################################
best_model = model.export_model()

print(best_model.summary())

PATH = "C:/ai5/_save/autokeras/"

best_model.save(PATH + 'kears70_autokeras1.h5')

# predict
y_predict1 = model.predict(x_test)
results1 = model.evaluate(x_test, y_test)
print('model result :', results1)

y_predict2 = best_model.predict(x_test)
# results2 = best_model.evaluate(x_test, y_test)
# print('best_model result :', results2)

print('time :', round(end_time - start_time, 2), 'sec')