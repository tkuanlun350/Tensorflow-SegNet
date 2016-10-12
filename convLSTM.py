"""
 Reference: https://github.com/iwyoo/ConvLSTMCell-tensorflow
"""
import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import init_ops
from tensorflow.python.ops.rnn_cell import LSTMStateTuple
from tensorflow.python.util import nest

# Future : Replace it with tensorflow.python.util.nest
import collections
import six

def _is_sequence(seq):
  return (isinstance(seq, collections.Sequence)
          and not isinstance(seq, six.string_types))

def ln(input, s, b, epsilon = 1e-5, max = 1000):
    """ Layer normalizes a 4D tensor along its second axis, which corresponds to batch """
    m, v = tf.nn.moments(input, [1,2,3], keep_dims=True) # for conv case ?
    normalised_input = (input - m) / tf.sqrt(v + epsilon)
    return normalised_input * s + b

class ConvGRUCell(rnn_cell.RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units, k_size=3, height=23, width=30, input_size=None, activation=tanh, initializer=None):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation
    self._initializer = initializer
    self._k_size = k_size
    self._height = height
    self._width = width

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def zero_state(self, batch_size=3, dtype=None):
    return tf.zeros([batch_size, self._height, self._width, self._num_units])

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
      with vs.variable_scope("Gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not update.
        r, u = array_ops.split(3, 2, _conv([inputs, state],
                                             2 * self._num_units, self._k_size, True, initializer=self._initializer))
        r, u = sigmoid(r), sigmoid(u)
      with vs.variable_scope("Candidate"):
        c = self._activation(_conv([inputs, r * state],
                                     self._num_units, self._k_size, True, initializer=self._initializer))
      new_h = u * state + (1 - u) * c
    return new_h, new_h

class ConvLSTMCell(rnn_cell.RNNCell):
  """ Convolutional LSTM network cell (ConvLSTM).
  The implementation is based on http://arxiv.org/abs/1506.04214.
   and BasicLSTMCell in TensorFlow.
   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell.py

   Future : Peephole connection will be added as the full LSTMCell
            implementation of TensorFlow.
  """
  def __init__(self, num_units, k_size=3, batch_size=4, height=23, width=30, input_size=None,
               use_peepholes=False, cell_clip=None,
               initializer=None, num_proj=None, proj_clip=None,
               num_unit_shards=1, num_proj_shards=1,
               forget_bias=1.0, state_is_tuple=False,
               activation=tanh):

    if not state_is_tuple:
      logging.warn(
          "%s: Using a concatenated state is slower and will soon be "
          "deprecated.  Use state_is_tuple=True." % self)
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated." % self)

    #self._use_peepholes = use_peepholes
    #self._cell_clip = cell_clip
    #self._initializer = initializer
    #self._num_proj = num_proj
    #self._num_unit_shards = num_unit_shards
    #self._num_proj_shards = num_proj_shards

    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation
    self._initializer = initializer
    self._k_size = k_size
    self._height = height
    self._width = width
    self._batch_size = batch_size

  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def zero_state(self, batch_size=4, dtype=None):
    return tf.zeros([batch_size, self._height, self._width, self._num_units*2])

  def __call__(self, inputs, state, scope=None):
    """Convolutional Long short-term memory cell (ConvLSTM)."""
    with vs.variable_scope(scope or type(self).__name__): # "ConvLSTMCell"
      if self._state_is_tuple:
        c, h = state
      else:
        c, h = array_ops.split(3, 2, state)

      # batch_size * height * width * channel
      concat = _conv([inputs, h], 4 * self._num_units, self._k_size, True, initializer=self._initializer)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = array_ops.split(3, 4, concat)

      new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) *
               self._activation(j))
      new_h = self._activation(new_c) * sigmoid(o)

      if self._state_is_tuple:
        new_state = LSTMStateTuple(new_c, new_h)
      else:
        new_state = array_ops.concat(3, [new_c, new_h])
      return new_h, new_state

class LNConvLSTMCell(rnn_cell.RNNCell):
  """ Convolutional LSTM network cell (ConvLSTM).
  The implementation is based on http://arxiv.org/abs/1506.04214.
   and BasicLSTMCell in TensorFlow.
   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell.py

   Future : Peephole connection will be added as the full LSTMCell
            implementation of TensorFlow.
  """
  def __init__(self, num_units, k_size=3, batch_size=4, height=23, width=30, input_size=None,
               use_peepholes=False, cell_clip=None,
               initializer=None, num_proj=None, proj_clip=None,
               num_unit_shards=1, num_proj_shards=1,
               forget_bias=1.0, state_is_tuple=False,
               activation=tanh):

    if not state_is_tuple:
      logging.warn(
          "%s: Using a concatenated state is slower and will soon be "
          "deprecated.  Use state_is_tuple=True." % self)
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated." % self)

    #self._use_peepholes = use_peepholes
    #self._cell_clip = cell_clip
    #self._initializer = initializer
    #self._num_proj = num_proj
    #self._num_unit_shards = num_unit_shards
    #self._num_proj_shards = num_proj_shards

    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation
    self._initializer = initializer
    self._k_size = k_size
    self._height = height
    self._width = width
    self._batch_size = batch_size

  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def zero_state(self, batch_size=4, dtype=None):
    return tf.zeros([batch_size, self._height, self._width, self._num_units*2])

  def __call__(self, inputs, state, scope=None):
    """Convolutional Long short-term memory cell (ConvLSTM)."""
    with vs.variable_scope(scope or type(self).__name__): # "ConvLSTMCell"
      if self._state_is_tuple:
        c, h = state
      else:
        c, h = array_ops.split(3, 2, state)
      s1 = vs.get_variable("s1", initializer=tf.ones([self._height, self._width, 4 * self._num_units]), dtype=tf.float32)
      s2 = vs.get_variable("s2", initializer=tf.ones([self._height, self._width, 4 * self._num_units]), dtype=tf.float32)
      # s3 = vs.get_variable("s3", initializer=tf.ones([self._batch_size, self._num_units]), dtype=tf.float32)

      b1 = vs.get_variable("b1", initializer=tf.zeros([self._height, self._width, 4 * self._num_units]), dtype=tf.float32)
      b2 = vs.get_variable("b2", initializer=tf.zeros([self._height, self._width, 4 * self._num_units]), dtype=tf.float32)
      # b3 = vs.get_variable("b3", initializer=tf.zeros([self._batch_size, self._num_units]), dtype=tf.float32)
      input_below_ = _conv([inputs], 4 * self._num_units, self._k_size, False, initializer=self._initializer, scope="out_1")
      input_below_ = ln(input_below_, s1, b1)
      state_below_ = _conv([h], 4 * self._num_units, self._k_size, False, initializer=self._initializer, scope="out_2")
      state_below_ = ln(state_below_, s2, b2)
      lstm_matrix = tf.add(input_below_, state_below_)

      i, j, f, o = array_ops.split(3, 4, lstm_matrix)

      # batch_size * height * width * channel
      # concat = _conv([inputs, h], 4 * self._num_units, self._k_size, True, initializer=self._initializer)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      # i, j, f, o = array_ops.split(3, 4, lstm_matrix)

      new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) *
               self._activation(j))
      new_h = self._activation(new_c) * sigmoid(o)

      if self._state_is_tuple:
        new_state = LSTMStateTuple(new_c, new_h)
      else:
        new_state = array_ops.concat(3, [new_c, new_h])
      return new_h, new_state

class MultiRNNCell(rnn_cell.RNNCell):

  def __init__(self, cells, state_is_tuple=False):
    """
      Stacked convLSTM , modified from ops.rnn_cell MultiRNNCell
    """
    if not cells:
      raise ValueError("Must specify at least one cell for MultiRNNCell.")
    self._cells = cells
    self._state_is_tuple = state_is_tuple
    self._num_units = cells[0].output_size
    if not state_is_tuple:
      if any(nest.is_sequence(c.state_size) for c in self._cells):
        raise ValueError("Some cells return tuples of states, but the flag "
                         "state_is_tuple is not set.  State sizes are: %s"
                         % str([c.state_size for c in self._cells]))

  @property
  def state_size(self):
    if self._state_is_tuple:
      return tuple(cell.state_size for cell in self._cells)
    else:
      return sum([cell.state_size for cell in self._cells])

  @property
  def output_size(self):
    return self._cells[-1].output_size

  def zero_state(self, batch_size=3, dtype=None, height=23, width=30):
    if self._state_is_tuple:
        return [tf.zeros(1, batch_size, height, width) for i in range(len(self._cells))]
    else:
        return tf.zeros([len(self._cells), batch_size, height, width, self._num_units*2])

  def __call__(self, inputs, state, scope=None):
    """Run this multi-layer cell on inputs, starting from state."""
    with vs.variable_scope(scope or type(self).__name__):  # "MultiRNNCell"
      cur_state_pos = 0
      cur_inp = inputs
      new_states = []
      for i, cell in enumerate(self._cells):
        with vs.variable_scope("Cell%d" % i):
          if self._state_is_tuple:
            if not nest.is_sequence(state):
              raise ValueError(
                  "Expected state to be a tuple of length %d, but received: %s"
                  % (len(self.state_size), state))
            cur_state = state[i]
          else:
            # print("STATE",state)
            """
            cur_state = array_ops.slice(
                state, [0, cur_state_pos], [-1, cell.state_size])
            """
            cur_state = array_ops.unpack(state)[i]
            # cur_state_pos += cell.state_size
          cur_inp, new_state = cell(cur_inp, cur_state)
          new_states.append(new_state)
    """
    new_states = (tuple(new_states) if self._state_is_tuple
                  else array_ops.concat(1, new_states))
    """
    new_states = array_ops.pack(new_states)
    return cur_inp, new_states

def _conv(args, output_size, k_size, bias=True, bias_start=0.0, initializer=None, scope=None):
  if args is None or (_is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not _is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 3.
  # (batch_size x height x width x arg_size)
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  height = shapes[0][1]
  width  = shapes[0][2]
  for shape in shapes:
    if len(shape) != 4:
      raise ValueError("Conv is expecting 3D arguments: %s" % str(shapes))
    if not shape[3]:
      raise ValueError("Conv expects shape[3] of arguments: %s" % str(shapes))
    if shape[1] == height and shape[2] == width:
      total_arg_size += shape[3]
    else :
      raise ValueError("Inconsistent height and width size in arguments: %s" % str(shapes))

  with vs.variable_scope(scope or "Conv"):
    kernel = vs.get_variable("Kernel", [k_size, k_size, total_arg_size, output_size], initializer=initializer)

    if len(args) == 1:
      res = tf.nn.conv2d(args[0], kernel, [1, 1, 1, 1], padding='SAME')
    else:
      res = tf.nn.conv2d(array_ops.concat(3, args), kernel, [1, 1, 1, 1], padding='SAME')

    if not bias: return res
    bias_term = vs.get_variable( "Bias", [output_size],
      initializer=init_ops.constant_initializer(bias_start))
  return res + bias_term
