import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/bin")
os.add_dll_directory("C:/tools/cuda/bin")
import tensorflow as tf
from PIL import Image

data = tf.data.TFRecordDataset('train.tfrecord')

data_feature_description = {
  'img_raw': tf.io.FixedLenFeature([], tf.string),
  'label': tf.io.FixedLenFeature([4], tf.int64)
}

def parse(example_proto):
  single = tf.io.parse_single_example(example_proto, data_feature_description)
  return tf.image.decode_png(single['img_raw']) / 255, tf.one_hot(single['label'], 10)

dataset = data.map(parse).shuffle(buffer_size = 6400).batch(640)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 72, 4)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(40, activation='softmax'))
model.add(tf.keras.layers.Reshape((4, 10)))

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(dataset, epochs=100)
