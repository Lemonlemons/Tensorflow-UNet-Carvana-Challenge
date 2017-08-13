import random
import requests

import os

import numpy as np
import tensorflow as tf

import csv
import sys
import json

import librosa
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2

def bytes_feature(value):
  '''
  Creates a TensorFlow Record Feature with value as a byte array.
  '''

  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
  '''
  Creates a TensorFlow Record Feature with value as a 64 bit integer.
  '''

  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def prepare_train_and_test_data(configs, args, input_size):
  train_path = configs['TF_RECORDS_TRAIN']
  test_path = configs['TF_RECORDS_TEST']
  meta_path = configs['TF_RECORDS_META']

  # If the data hasn't been preprocessed, then do it now.
  if not os.path.exists(train_path) \
      or not os.path.exists(test_path) \
      or not os.path.exists(meta_path):
    print('Preparing Training and Testing Data')

    df_train = pd.read_csv(args.data_location + '/train_masks.csv')
    ids_train = df_train['img'].map(lambda s: s.split('.')[0])

    ids_train_split, ids_test_split = train_test_split(ids_train, test_size=0.2, random_state=43)

    # Write the training set.
    training_count = prepare_unet_tfrecord(train_path, args, ids_train_split, input_size)

    # Write the testing set.
    test_count = prepare_unet_tfrecord(test_path, args, ids_test_split, input_size)

    with open(meta_path, 'w') as OUTPUT:
      OUTPUT.write('{},{}'.format(training_count, test_count))

  print('preprocessing completed')

def get_stats(configs):
  meta_file = configs['TF_RECORDS_META']
  with open(meta_file, 'r') as INPUT:
    META_DATA = INPUT.readline()
    tuple = [
      float(DATA_POINT) for DATA_POINT in META_DATA.split(',')
    ]
  return tuple

# Big thanks to Peter Giannakopoulos!
def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
  if np.random.random() < u:
    height, width, channel = image.shape

    angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
    scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
    aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
    sx = scale * aspect / (aspect ** 0.5)
    sy = scale / (aspect ** 0.5)
    dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
    dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

    cc = np.math.cos(angle / 180 * np.math.pi) * sx
    ss = np.math.sin(angle / 180 * np.math.pi) * sy
    rotate_matrix = np.array([[cc, -ss], [ss, cc]])

    box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
    box1 = box0 - np.array([width / 2, height / 2])
    box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)
    image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                borderValue=(
                                    0, 0,
                                    0,))
    mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                               borderValue=(
                                   0, 0,
                                   0,))

  return image, mask

def randomHorizontalFlip(image, mask, u=0.5):
  if np.random.random() < u:
    image = cv2.flip(image, 1)
    mask = cv2.flip(mask, 1)

  return image, mask

def prepare_unet_tfrecord(set_path, args, ids, input_size):

  writer = tf.python_io.TFRecordWriter(set_path)

  count = len(ids)

  for id in ids:
    img = cv2.imread(args.data_location + '/train/{}.jpg'.format(id))
    img = cv2.resize(img, (input_size, input_size))
    mask = cv2.imread(args.data_location + '/train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (input_size, input_size))

    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.0625, 0.0625),
                                       scale_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32) / 255
    mask = np.array(mask, np.float32) / 255

    print("processed: " + id)

    # Write the final input frames and binary_mask to disk.
    example = tf.train.Example(features=tf.train.Features(feature={
      'images': bytes_feature(img.flatten().tostring()),
      'masks': bytes_feature(mask.flatten().tostring())
    }))
    writer.write(example.SerializeToString())

  writer.close()
  return count
