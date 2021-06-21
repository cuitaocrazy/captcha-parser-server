import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/bin")
os.add_dll_directory("C:/tools/cuda/bin")
import json
from pathlib import Path
from PIL import Image
import tensorflow as tf

rootDir = Path('./')
jsonPath = (rootDir / 'data.json')
imgsSubPath = rootDir / 'captcha-imgs'
tfrecordPath = rootDir / 'tfrecord.data'

with jsonPath.open() as jf, tf.io.TFRecordWriter('train.tfrecord') as writer:
  data = json.load(jf)
  for meta in data:
    if len(meta['label']) != 4:
      continue
    img = open((imgsSubPath / meta['img']).resolve(), 'rb').read()
    label = [int(x) for x in meta['label']]
    example = tf.train.Example(features=tf.train.Features(feature={
      "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
      "label": tf.train.Feature(int64_list=tf.train.Int64List(value=label))}))
    writer.write(example.SerializeToString())
