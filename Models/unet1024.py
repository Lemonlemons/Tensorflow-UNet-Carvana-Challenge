from .base_model import *
from .layers import *

import numpy as np
import tensorflow as tf

class DNN1Model(BaseModel):

  def build(self, stats):
    # image shape
    self.image_input_shape = (1024, 1024, 3)
    # mask shape
    self.mask_input_shape = (1024, 1024, 1)
    # number of classes
    self.num_classes = 1
    # batch size
    if self.is_testing or self.is_training:
      self.batch_size = 16
    else:
      self.batch_size = 1
    self.learning_rate = 1e-4

    # Build our dataflow graph.
    self.GRAPH = tf.Graph()
    with self.GRAPH.as_default():
      self.IMAGES, self.MASKS = self.read_inputs([self.input_file], images_shape=self.image_input_shape, masks_shape=self.mask_input_shape,
        batch_size=self.batch_size, capacity=10000, min_after_dequeue=self.batch_size,
        num_threads=self.input_pipeline_threads, is_training=(self.is_training or self.is_testing)
      )

      training_set_count, test_set_count = stats

      print('training set count: ' + str(training_set_count))
      print('test set count: ' + str(test_set_count))

      # make them into "magnitude spectograms"
      self.CORRECTED_SPECTOGRAMS = tf.abs(self.SPECTOGRAMS)

      # Build feedforward layers.
      # This first layer is supposed to be a "locally_connected" layer however tensorflow doesn't have an implementation of that.
      with tf.variable_scope('fully_connected_1') as scope:
        H_1 = fc_dropout_layer(self.CORRECTED_SPECTOGRAMS, np.prod(self.spectograms_shape), self.n_cells, scope, self.keep_prob, is_training=self.is_training, act_func=selu)
      with tf.variable_scope('fully_connected_2') as scope:
        H_2 = fc_dropout_layer(H_1, self.n_cells, self.n_cells, scope, self.keep_prob, is_training=self.is_training, act_func=selu)
      with tf.variable_scope('fully_connected_3') as scope:
        H_3 = fc_dropout_layer(H_2, self.n_cells, self.n_cells, scope, self.keep_prob, is_training=self.is_training, act_func=selu)
      with tf.variable_scope('fully_connected_4') as scope:
        self.LOGITS = fc_layer(H_3, self.n_cells, self.n_classes, scope, is_training=self.is_training)
      self.Y = tf.nn.sigmoid(self.LOGITS)

      print(self.LOGITS.shape)

      # Compute the cross entropy loss.
      self.COST = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=self.MASKS, logits=self.LOGITS
      ))

      tf.summary.scalar("cost", self.COST)

      # Compute the accuracy
      self.ACCURACY = tf.subtract(tf.cast(1.0, tf.float64), tf.reduce_mean(tf.abs(tf.subtract(self.Y, self.MASKS))))
      tf.summary.scalar("accuracy", self.ACCURACY)

      # Compute gradients.
      OPTIMIZER = tf.train.AdamOptimizer(self.learning_rate)

      GRADIENTS = OPTIMIZER.compute_gradients(self.COST)

      # Apply gradients.
      self.APPLY_GRADIENT_OP = OPTIMIZER.apply_gradients(GRADIENTS)

      # Add histograms for gradients to our TensorBoard logs.
      for GRADIENT, VAR in GRADIENTS:
        if GRADIENT is not None:
          tf.summary.histogram('{}/gradients'.format(VAR.op.name), GRADIENT)

      # Collect the TensorBoard summaries.
      self.SUMMARIES_OP = tf.summary.merge_all()