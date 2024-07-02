import os
# This is used to preproccess and import videos
import cv2
import tensorflow as tf
import numpy as np
from typing import List
from matplotlib import pyplot as plt
import imageio

import matplotlib
matplotlib.use('Agg')

# library to download from google drive...
import gdown

# Function to load the alignment based on the path
def load_alignments(path:str) -> List[str]:
  with open(path, 'r') as f:
    lines = f.readlines()
  tokens = []
  for line in lines:
    line = line.split()
    if line[2] != 'sil':
      tokens = [*tokens, ' ', line[2]]
  return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

# Function to load video frames
def load_video(path:str) -> List[float]:
  # Load the video and get frame per frame
  cap = cv2.VideoCapture(path)
  frames = []
  for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    ret, frame = cap.read()
    # Change frame to grayscale
    frame = tf.image.rgb_to_grayscale(frame)
    # Isolate only the mouth at the frame
    frames.append(frame[190:236,80:220,:])
  cap.release()

  mean = tf.math.reduce_mean(frames)
  std = tf.math.reduce_std(tf.cast(frames, tf.float32))
  return tf.cast((frames - mean), tf.float32) / std

# Function to load both alignement and video frame
def load_data(path:str):
    path = bytes.decode(path.numpy())
    # Grabing the filename by spliting the path and getting the last element on array
    file_name = path.split('/')[-1].split('.')[0]
    video_path = os.path.join('data', 's1', f'{file_name}.mpg')
    alignment_path = os.path.join('data', 'alignments', 's1', f'{file_name}.align')
    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)

    return frames, alignments

def mappable_function(path:str) -> List[str]:
  result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
  return result

# Here we are prevent exponential memory grow
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  pass

# Checks if the folder data already exists, if not download the dataset of videos
if os.path.exists('./data') == False:
  url = 'https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL'
  output = 'data.zip'
  gdown.download(url, output, quiet=False)
  gdown.extractall('data.zip')


# All caracters that we are going to use
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?123456789 "]

char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
  vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# print(
#   f"The vocabulary is: {char_to_num.get_vocabulary()}"
#   f"(size = {char_to_num.vocabulary_size()})"
# )

test_path = './data/s1/bbal6n.mpg'
tf.convert_to_tensor(test_path).numpy().decode('utf-8').split('/')[-1].split('.')[0]
frames, alignments = load_data(tf.convert_to_tensor(test_path))

plt.figure(figsize=(10,10))
test = plt.imshow(frames[40])
plt.savefig('test.png')

# print(tf.strings.reduce_join([bytes.decode(x) for x in num_to_char(alignments.numpy()).numpy()]))

# Creating data pipeline

data = tf.data.Dataset.list_files('./data/s1/*.mpg')
print(data)

# return each file path
# data.as_numpy_iterator().next() 

data = data.shuffle(500)
data = data.map(mappable_function)

# Fill with 75 frames and 40 positions with 0 pad, why do this?
data = data.padded_batch(2, padded_shapes=([75,None,None,None],[40]))
data = data.prefetch(tf.data.AUTOTUNE)

test = data.as_numpy_iterator()
val = test.next()
print(val[0][1])

frames_to_save = (val[0][1] * 255).astype(np.uint8)

# Remova qualquer dimensão extra
frames_to_save = frames_to_save.squeeze()

# Verifique o tipo de dados e a forma
print(f"Shape: {frames_to_save.shape}, Dtype: {frames_to_save.dtype}")

imageio.mimsave('./animation.gif', frames_to_save, fps=10)

# 0: videos, 0: first video, 0: first frame
# plt.imshow(val[0][0][0])
# plt.savefig('test.png')
# plt.show()

print(tf.strings.reduce_join([num_to_char(word) for word in val[1][0]]), 'estou aq')