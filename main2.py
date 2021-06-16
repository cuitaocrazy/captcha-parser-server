import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import json
import cv2
from PIL import Image

rootDir = '/Users/cuitao/projects/p/captcha-img-download/'
jsonPath = rootDir + 'data.json'
imgsSubPath = 'public/captcha-imgs/'

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
# with open(jsonPath) as js:
#   data = json.load(js)
#   data = [data[0], data[1]]
#   labels = [[tf.keras.utils.to_categorical(l, 10) for l in m['label']][0] for m in data]
#   # print(rootDir+imgsSubPath+data[0]['img'])
#   imgs = [imgData(rootDir+imgsSubPath+m['img']) for m in data]
#   # imgs = imgData(rootDir+imgsSubPath+data[0]['img'])

with open(jsonPath) as jf:
  data = json.load(jf)
  # data = [data[0], data[1]]
  labels = np.array([np.array([tf.keras.utils.to_categorical(l, 10) for l in m['label']]) for m in data])
  imgs = np.array([imgData(rootDir+imgsSubPath+m['img'])/255 for m in data])
  # print(imgs)

x_train = imgs
y_train = labels

print(labels)
model = tf.keras.models.Sequential()
# ([
#   tf.keras.layers.Flatten(input_shape=(28, 72)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(40, activation='softmax'),
#   tf.keras.layers.Reshape((4, 10))
# ])

model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 72, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(40, activation='softmax'))
model.add(tf.keras.layers.Reshape((4, 10)))
print(model.inputs[0])
model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100)

model.save('cnn-ret')