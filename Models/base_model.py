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
import cv2
from tqdm import tqdm


class BaseModel(object):
  def __init__(self, args, stats, configs):
    training_count, testing_count = stats
    self.mode = args.phase
    self.delete_old = args.delete_old == 'True'
    self.model = args.model
    self.model_file = 'Results/' + self.model + '/unet.ckpt'
    self.is_training = self.mode == 'train'
    self.is_testing = self.mode == 'test'
    if self.is_training:
      self.input_file = configs['TF_RECORDS_TRAIN']
    else:
      self.input_file = configs['TF_RECORDS_TEST']
    self.num_gpus = int(args.num_gpus)

    self.batch_size = configs['BATCH_SIZE']

    self.num_batches = int(np.ceil(float(training_count)/float(self.batch_size)))
    self.num_epochs = configs['NUMBER_OF_EPOCHS']
    self.stagnent_epochs_threshold = 6
    self.stagnent_lr_factor = 0.1
    self.lr = configs['LEARNING_RATE']

    self.input_pipeline_threads = 1
    self.graph_config = tf.ConfigProto(allow_soft_placement=True,
                                       log_device_placement=False,
                                       inter_op_parallelism_threads=20,
                                       intra_op_parallelism_threads=8)

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
      BEST_DICE_LOSS = 0.0
      STAGNENT_EPOCHS = 0
      LEARNING_RATE = self.lr
      for EPOCH in range(self.num_epochs):
        DURATION = 0
        ERROR = 0.0
        START_TIME = time.time()
        ACCS = []
        for MINI_BATCH in range(self.num_batches):
          _, SUMMARIES, COST_VAL, DICE_LOSS = SESSION.run([
            self.APPLY_GRADIENT_OP, self.SUMMARIES_OP, self.COST, self.DICE_LOSS
          ], feed_dict={self.learning_rate: LEARNING_RATE})
          ERROR += COST_VAL
          GLOBAL_STEP += 1
          ACCS.append(DICE_LOSS)
          if len(ACCS) >= 3:
            AVG_DICE_LOSS = np.mean(ACCS)
            if AVG_DICE_LOSS > BEST_DICE_LOSS:
              BEST_DICE_LOSS = AVG_DICE_LOSS
              print('Saving Session, loss: ' + str(BEST_DICE_LOSS))
              GRAPH_SAVER.save(SESSION, self.model_file)

        # Write the summaries to disk.
        SUMMARY_WRITER.add_summary(SUMMARIES, EPOCH)
        DURATION += time.time() - START_TIME
        TOTAL_DURATION += DURATION
        # Update the console.
        print('Epoch %d: loss = %.4f (%.3f sec), dice loss = %.8f' % (EPOCH, ERROR, DURATION, DICE_LOSS))
        # Check for stagnent epochs
        if DICE_LOSS > BEST_DICE_LOSS:
          STAGNENT_EPOCHS = 0
        else:
          STAGNENT_EPOCHS += 1
        # Check if there is a learning plateau and the LR should be decreased
        if STAGNENT_EPOCHS >= self.stagnent_epochs_threshold:
          print("Reducing learning rate")
          LEARNING_RATE = LEARNING_RATE * self.stagnent_lr_factor
          STAGNENT_EPOCHS = 0
        # Check if loss is good enough to end early
        if EPOCH == self.num_epochs or DICE_LOSS > 0.997:
          print(
            'Done training for %d epochs. (%.3f sec) total steps %d' % (EPOCH, TOTAL_DURATION, GLOBAL_STEP)
          )
          break
      print('Training Done!')
      COORDINATOR.request_stop()
      COORDINATOR.join(THREADS)

  # get test accuracies of models
  def test(self):
    print('testing Model')
    with tf.Session(graph=self.GRAPH, config=self.graph_config) as SESSION:
      COORDINATOR = tf.train.Coordinator()
      SESSION.run(tf.global_variables_initializer())
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

  # https://www.kaggle.com/stainsby/fast-tested-rle
  def run_length_encode(self, mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle

  # create a Kaggle submission
  def create_submission(self, configs):
    print('preparing Kaggle submission')
    df_test = pd.read_csv('input/sample_submission.csv')
    ids_test = df_test['img'].map(lambda s: s.split('.')[0])
    orig_width = 1918
    orig_height = 1280
    threshold = 0.5
    names = []
    for id in ids_test:
      names.append('{}.jpg'.format(id))
    rles = []
    with tf.Session(graph=self.GRAPH, config=self.graph_config) as SESSION:
      # restore the session
      GRAPH_WRITER = tf.train.Saver()
      GRAPH_WRITER.restore(SESSION, self.model_file)

      for start in tqdm(range(0, len(ids_test), self.batch_size)):
        x_batch = []
        end = min(start + self.batch_size, len(ids_test))
        ids_test_batch = ids_test[start:end]
        for id in ids_test_batch.values:
          img = cv2.imread('input/test/{}.jpg'.format(id))
          img = cv2.resize(img, (configs['INPUT_SIZE'], configs['INPUT_SIZE']))
          x_batch.append(img)
        x_batch = np.array(x_batch, np.float32) / 255
        x_batch = np.reshape(x_batch, [self.batch_size, np.prod(self.image_input_shape)])
        preds = SESSION.run(self.Y, feed_dict={self.images: x_batch})
        preds = np.squeeze(preds, axis=-1)
        for pred in preds:
          prob = cv2.resize(pred, (orig_width, orig_height))
          mask = prob > threshold
          rle = self.run_length_encode(mask)
          rles.append(rle)
    print("Generating submission file...")
    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv('submission.csv.gz', index=False, compression='gzip')

  # custom loss functions
  def dice_loss(self, y_true, y_pred):
    smooth = 1.
    y_true_f = tf.contrib.layers.flatten(y_true)
    y_pred_f = tf.contrib.layers.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1) + smooth)

  def bce_dice_loss(self, y_true, y_pred, y_pred_pre_sigmoid):
    # return tf.contrib.keras.backend.binary_crossentropy(y_true, y_pred) + (1 - self.dice_loss(y_true, y_pred))
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred_pre_sigmoid) + (1 - self.dice_loss(y_true, y_pred))

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
          capacity=capacity, min_after_dequeue=min_after_dequeue,
          num_threads=num_threads,
        )
      else:
        images_shape = [batch_size] + images_shape
        self.images = tf.placeholder(tf.float32, shape=images_shape)
        masks_shape = [batch_size] + masks_shape
        self.masks = tf.placeholder(tf.float32, shape=masks_shape)

      return self.images, self.masks