import tensorflow as tf
import numpy as np
from PIL import Image

def imgData(path):
  with Image.open(path) as img:
    img = img.convert('F')
    data = np.array(img)

    for i, row in enumerate(data):
      if i == 0 or i == img.size[1] - 1:
        for j in range(img.size[0]):
          row[j] = 255
      else:
        row[0] = 255
        row[img.size[0] - 1] = 255
      for j, col in enumerate(row):
        row[j] = 255 if row[j] > 180 else 0
  return np.atleast_3d(data)

vd = imgData('test.png')/255

img = np.expand_dims(vd, 0)

new_model = tf.keras.models.load_model('cnn-ret')

for _ in range(10):
  predictions_single =new_model.predict(img)
  _label = tf.math.argmax(predictions_single, axis=-1)
  label = "".join([str(n.numpy()) for n in _label[0]])
  print(label)
