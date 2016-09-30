from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell

class ConvGRUCell(RNNCell):
  """
  conv GRU cell
  input: [batch, _w, _h, _channels]
  kernel_U: [kernel_size, kernel_size, in_channel(input_size), out_channel(output_size)]
  kernel_W: [kernel_size, kernel_size, in_channel(state_size = output_size), out_channel(output_size)]
  """
  def __init__(self, shape=None, input_size=None, output_size=None, kernel_size=None, activation=tanh):
    if input_size is None or output_size is None or kernel_size is None:
      logging.warn("%s: Some parameters are missing.", self)
    self._num_units = shape
    self._activation = activation
    self._input_size = input_size
    self._output_size = output_size
    self._kernel_size = kernel_size

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units
    
  def zero_state(self, batch_size, dtype):
    return tf.zeros(self._num_units, tf.float32)
  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (ConvGRU) with nunits cells."""

    kernel_size = self._kernel_size
    input_size = self._input_size
    output_size = self._output_size

    with vs.variable_scope("ConvGRU"):  # "GRUCell"
      with vs.variable_scope("Gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not update.
        kernel_U_R = vs.get_variable("K_U_R", [kernel_size, kernel_size, input_size, output_size])
        kernel_W_R = vs.get_variable("K_W_R", [kernel_size, kernel_size, output_size, output_size])
        conv_U_R = tf.nn.conv2d(inputs, kernel_U_R, [1, 1, 1, 1], padding='SAME')
        conv_W_R = tf.nn.conv2d(state, kernel_W_R, [1, 1, 1, 1], padding='SAME')
        r = sigmoid(conv_U_R + conv_W_R)

        kernel_U_U = vs.get_variable("K_U_U", [kernel_size, kernel_size, input_size, output_size])
        kernel_W_U = vs.get_variable("K_W_U", [kernel_size, kernel_size, output_size, output_size])
        conv_U_U = tf.nn.conv2d(inputs, kernel_U_U, [1, 1, 1, 1], padding='SAME')
        conv_W_U = tf.nn.conv2d(state, kernel_W_U, [1, 1, 1, 1], padding='SAME')
        u = sigmoid(conv_U_U + conv_W_U)
      with vs.variable_scope("Candidate"):
        kernel_U_H = vs.get_variable("K_U_H", [kernel_size, kernel_size, input_size, output_size])
        kernel_W_H = vs.get_variable("K_W_H", [kernel_size, kernel_size, output_size, output_size])
        conv_U_H = tf.nn.conv2d(inputs, kernel_U_H, [1, 1, 1, 1], padding='SAME')
        conv_W_H = tf.nn.conv2d(tf.mul(state , r), kernel_W_H, [1, 1, 1, 1], padding='SAME')
        c = tanh(conv_U_H + conv_W_H)
      new_h = tf.mul(u , state) + tf.mul((1 - u) , c)
    return new_h, new_h
