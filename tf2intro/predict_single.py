import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from utils import plot_image, plot_value_array

fashion_mnist = keras.datasets.fashion_mnist

_, (test_images, test_labels) = fashion_mnist.load_data()

test_images = test_images / 255.0

model = keras.models.load_model('clf_fmnist.h5')

i_target = 100

img = test_images[i_target]
print(img.shape)

img = (np.expand_dims(img,0))
print(img.shape)

predictions_single = model.predict(img)
print(predictions_single)
print(np.argmax(predictions_single[0]))

plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(0, predictions_single, [test_labels[i_target]], [test_images[i_target]])
plt.subplot(1,2,2)
plot_value_array(0, predictions_single, [test_labels[i_target]])
plt.show()

