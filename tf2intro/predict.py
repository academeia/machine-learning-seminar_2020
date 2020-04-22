import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

from utils import plot_image, plot_value_array

fashion_mnist = keras.datasets.fashion_mnist

_, (test_images, test_labels) = fashion_mnist.load_data()

test_images = test_images / 255.0

model = keras.models.load_model('clf_fmnist.h5')
predictions = model.predict(test_images)

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

