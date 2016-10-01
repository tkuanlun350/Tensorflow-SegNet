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

# Future : Replace it with tensorflow.python.util.nest
import collections
import six
def _is_sequence(seq):
  return (isinstance(seq, collections.Sequence)
          and not isinstance(seq, six.string_types))

class ConvLSTMCell(rnn_cell.RNNCell):
  """ Convolutional LSTM network cell (ConvLSTM).
  The implementation is based on http://arxiv.org/abs/1506.04214.
   and BasicLSTMCell in TensorFlow.
   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell.py

   Future : Peephole connection will be added as the full LSTMCell
            implementation of TensorFlow.
  """
  def __init__(self, num_units, input_size=None,
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

  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def zero_state(self, batch_size=3, dtype=None, height=15, width=15):
    return tf.zeros([batch_size, height, width, self._num_units*2])

  def __call__(self, inputs, state, k_size=7, scope=None):
    """Convolutional Long short-term memory cell (ConvLSTM)."""
    with vs.variable_scope(scope or type(self).__name__): # "ConvLSTMCell"
      if self._state_is_tuple:
        c, h = state
      else:
        c, h = array_ops.split(3, 2, state)

      # batch_size * height * width * channel
      concat = _conv([inputs, h], 4 * self._num_units, k_size, True)

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

def _conv(args, output_size, k_size, bias=True, bias_start=0.0, scope=None):
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
    kernel = vs.get_variable("Kernel", [k_size, k_size, total_arg_size, output_size])

    if len(args) == 1:
      res = tf.nn.conv2d(args[0], kernel, [1, 1, 1, 1], padding='SAME')
    else:
      res = tf.nn.conv2d(array_ops.concat(3, args), kernel, [1, 1, 1, 1], padding='SAME')

    if not bias: return res
    bias_term = vs.get_variable( "Bias", [output_size],
      initializer=init_ops.constant_initializer(bias_start))
  return res + bias_term
