import os
# This is used to preproccess and import videos
import cv2
import tensorflow as tf
import numpy as np
from typing import List
from matplotlib import pyplot as plt
import imageio
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

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


# Learning rate scheduler?
def scheduler(epoch, lr):
  if epoch < 30:
    return lr
  else:
    return lr * tf.math.exp(-0.1)
  
# From Keras.io
def CTCLoss(y_true, y_pred):
  batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
  input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
  label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

  input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
  label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

  loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
  return loss

class ProduceExample(tf.keras.callbacks.Callback): 
  def __init__(self, dataset) -> None: 
    self.dataset = dataset.as_numpy_iterator()
    
  def on_epoch_end(self, epoch, logs=None) -> None:
    data = self.dataset.next()
    print(data)
    yhat = self.model.predict(data[0])
    decoded = tf.keras.backend.ctc_decode(yhat, [75], greedy=False)[0][0].numpy()
    for x in range(len(yhat)):           
      print('Original:', tf.strings.reduce_join(num_to_char(data[1][x])).numpy().decode('utf-8'))
      print('Prediction:', tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8'))
      print('~'*100)

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
# print(data)

# return each file path
# data.as_numpy_iterator().next() 

data = data.shuffle(500, reshuffle_each_iteration=False)
data = data.map(mappable_function)

# Fill with 75 frames and 40 positions with 0 pad, why do this?
data = data.padded_batch(1, padded_shapes=([75,None,None,None],[40]))
data = data.prefetch(tf.data.AUTOTUNE)

# Added for split 
train = data.take(1)
test = data.skip(1)

sample = data.as_numpy_iterator()

val = sample.next()
# print(val[0][1])

frames_to_save = (val[0] * 255).astype(np.uint8)

# Remova qualquer dimensão extra
frames_to_save = frames_to_save.squeeze()

# Verifique o tipo de dados e a forma
# print(f"Shape: {frames_to_save.shape}, Dtype: {frames_to_save.dtype}")

imageio.mimsave('./animation.gif', frames_to_save, fps=10)

# 0: videos, 0: first video, 0: first frame
# plt.imshow(val[0][0][0])
# plt.savefig('test.png')
# plt.show()

# print(tf.strings.reduce_join([num_to_char(word) for word in val[1][0]]), 'estou aq')

# Design the Deep Neural Network

model = Sequential()

# What is Conv3D

# This is the same shape as the video data.as_numpy_iterator().next()[0][0].shape
# First layer
model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))

model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

# Second layer
model.add(Conv3D(256, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

# Third layer
model.add(Conv3D(75, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))


# print("Model input shape:", model.input_shape)
# print("Model output shape:", model.output_shape)

model.add(Reshape((75, -1)))

model.add(TimeDistributed(Flatten()))

# Bidirection, LSTM??
model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
model.add(Dropout(.5))

model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
model.add(Dropout(.5))

model.add(Dense(char_to_num.vocabulary_size()+1, kernel_initializer='he_normal', activation='softmax'))

print(model.summary())

yhat = model.predict(val[0])

# See what our model is predicting
print(tf.strings.reduce_join([num_to_char(tf.argmax(x)) for x in yhat[0]]))
print('isso que eu quero...')

# print(model.input_shape)

# print(model.output_shape)

# Let's train
# Let's train

model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)

checkpoint_callback = ModelCheckpoint(os.path.join('models','checkpoint.weights.h5'), monitor='loss', save_weights_only=True)

schedule_callback = LearningRateScheduler(scheduler)
example_callback = ProduceExample(test)

model.fit(train, validation_data=test, epochs=1, callbacks=[checkpoint_callback, schedule_callback, example_callback])
