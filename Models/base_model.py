import csv

import tensorflow as tf
import time
import shutil
import os
import numpy as np
from numpy import dot
from numpy.linalg import norm
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants
import boto3
from .preprocessing import *
import requests


class BaseModel(object):
  def __init__(self, args, stats, input_file):
    training_count, testing_count = stats
    self.mode = args.phase
    self.delete_old = args.delete_old == 'True'
    self.model = args.model
    self.model_file = 'Results/' + self.model + '/unet.ckpt'
    self.is_training = self.mode == 'train'
    self.is_testing = self.mode == 'test'
    self.input_file = input_file
    if self.is_testing or self.is_training:
      self.batch_size = 4
    else:
      self.batch_size = 1
    self.num_batches = 9 # int(training_count/self.batch_size)
    self.num_epochs = 1500
    self.learning_rate = 1e-5

    self.input_pipeline_threads = 2
    self.graph_config = tf.ConfigProto(allow_soft_placement=True,
                                       log_device_placement=False,
                                       inter_op_parallelism_threads=5,
                                       intra_op_parallelism_threads=2)

    print('building Model')
    self.build(stats)

  def build(self, stats):
    raise NotImplementedError

  # training the chosen model
  def train(self):
    print('training Model')
    # Start training the model.
    # this session is for multi-gpu training
    with tf.Session(graph=self.GRAPH, config=self.graph_config) as SESSION:
      # Create Coordinator
      COORDINATOR = tf.train.Coordinator()

      # Initialize all the variables.
      SESSION.run(tf.global_variables_initializer())

      if self.delete_old:
        # remove old tensorboard and models files:
        shutil.rmtree('Results/'+self.model)
        os.makedirs('Results/'+self.model)
      else:
        # restore the session
        GRAPH_WRITER = tf.train.Saver()
        GRAPH_WRITER.restore(SESSION, self.model_file)

      shutil.rmtree('Tensorboard/' + self.model)
      os.makedirs('Tensorboard/' + self.model)

      # Start Queue Runners
      THREADS = tf.train.start_queue_runners(SESSION, COORDINATOR)
      # Create a tensorflow summary writer.
      SUMMARY_WRITER = tf.summary.FileWriter('Tensorboard/'+self.model, graph=self.GRAPH)
      # Create a tensorflow graph writer.
      GRAPH_SAVER = tf.train.Saver(tf.global_variables())

      TOTAL_DURATION = 0.0
      GLOBAL_STEP = 0
      BEST_COST = 0.0
      for EPOCH in range(self.num_epochs):
        DURATION = 0
        ERROR = 0.0
        ACC = 0.0
        START_TIME = time.time()
        for MINI_BATCH in range(self.num_batches):
          _, SUMMARIES, COST_VAL, DICE_LOSS = SESSION.run([
            self.APPLY_GRADIENT_OP, self.SUMMARIES_OP, self.COST, self.DICE_LOSS
          ])
          ERROR += COST_VAL
          GLOBAL_STEP += 1

        # Write the summaries to disk.
        SUMMARY_WRITER.add_summary(SUMMARIES, EPOCH)
        DURATION += time.time() - START_TIME
        TOTAL_DURATION += DURATION
        # Update the console.
        print('Epoch %d: loss = %.4f (%.3f sec), dice loss = %.8f' % (EPOCH, ERROR, DURATION, DICE_LOSS))
        if EPOCH != 0 and DICE_LOSS > BEST_COST:
          BEST_COST = DICE_LOSS
          print('Saving Session')
          GRAPH_SAVER.save(SESSION, self.model_file)
        # if EPOCH == self.num_epochs or ERROR < 0.005:
        #   print(
        #     'Done training for %d epochs. (%.3f sec) total steps %d' % (EPOCH, TOTAL_DURATION, GLOBAL_STEP)
        #   )
        #   break
      print('Training Done!')
      COORDINATOR.request_stop()
      COORDINATOR.join(THREADS)

  # get test accuracies of models
  def test(self):
    print('testing Model')
    with tf.Session(graph=self.GRAPH, config=self.graph_config) as SESSION:
      COORDINATOR = tf.train.Coordinator()
      THREADS = tf.train.start_queue_runners(SESSION, COORDINATOR)

      # restore the session
      GRAPH_WRITER = tf.train.Saver()
      GRAPH_WRITER.restore(SESSION, self.model_file)

      for EPOCH in range(10):
        DICE_LOSS = SESSION.run(self.DICE_LOSS)
        # Update the console.
        print('Epoch %d: dice loss = %.8f' % (EPOCH, DICE_LOSS))
      COORDINATOR.request_stop()
      COORDINATOR.join(THREADS)

  # custom loss functions
  def dice_loss(self, y_true, y_pred):
    smooth = 1.
    y_true_f = tf.contrib.layers.flatten(y_true)
    y_pred_f = tf.contrib.layers.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1) + smooth)

  def bce_dice_loss(self, y_true, y_pred):
    return tf.contrib.keras.backend.binary_crossentropy(y_true, y_pred) + (1 - self.dice_loss(y_true, y_pred))

  # read inputs from tfrecords
  def read_inputs(self, file_paths, images_shape, masks_shape, batch_size=64,
                  capacity=1000, min_after_dequeue=900, num_threads=2, is_training=True):

    with tf.name_scope('input'):
      # if training we use an input queue otherwise we use placeholders
      if is_training:
        # Create a file name queue.
        filename_queue = tf.train.string_input_producer(file_paths)
        reader = tf.TFRecordReader()
        # Read an example from the TFRecords file.
        _, example = reader.read(filename_queue)
        features = tf.parse_single_example(example, features={
          'images': tf.FixedLenFeature([], tf.string),
          'masks': tf.FixedLenFeature([], tf.string)
        })
        # Decode sample
        image = tf.decode_raw(features['images'], tf.float32)
        image.set_shape(images_shape)
        mask = tf.decode_raw(features['masks'], tf.float32)
        mask.set_shape(masks_shape)

        self.images, self.masks = tf.train.shuffle_batch(
          [image, mask], batch_size=batch_size,
          capacity=capacity, min_after_dequeue=min_after_dequeue, num_threads=num_threads,
        )
      else:
        images_shape = [None] + images_shape
        self.images = tf.placeholder(tf.float64, shape=images_shape)
        masks_shape = [None] + masks_shape
        self.masks = tf.placeholder(tf.float64, shape=masks_shape)

      return self.images, self.masks