import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops

import os, sys
import numpy as np
import math
from datetime import datetime
import time
import Image
from math import ceil
from tensorflow.python.ops import gen_nn_ops
import skimage
import skimage.io
# modules
import Utils
import convGRU
import convLSTM
from Inputs import *

LOSS_WEIGHT = np.array([
  0.2595,
  0.1826,
  4.5640,
  0.1417,
  0.9051,
  0.3826,
  9.6446,
  1.8418,
  0.6823,
  6.2478,
  7.3614,
]) # class 0~10

"""
LOSS_WEIGHT = np.asarray([
  0.0616944799702,
  3.89114328416,
  0.718496198987,
  3.24645148591,
  1.64418466389
]) # class 0~5
"""
@ops.RegisterGradient("MaxPoolWithArgmax")
def _MaxPoolWithArgmaxGrad(op, grad, unused_argmax_grad):
  return gen_nn_ops._max_pool_grad(op.inputs[0],
                                   op.outputs[0],
                                   grad,
                                   op.get_attr("ksize"),
                                   op.get_attr("strides"),
                                   padding=op.get_attr("padding"),
                                   data_format='NHWC')

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

INITIAL_LEARNING_RATE = 0.0001      # Initial learning rate.
EVAL_BATCH_SIZE = 1
BATCH_SIZE = 4
SEQUENCE_LENGTH = 3
# for CamVid
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 480
# IMAGE_DEPTH = 3
# for miccai
IMAGE_DEPTH = 3

NUM_CLASSES = 11
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 367
#NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 101
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 233
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1
TEST_ITER = NUM_EXAMPLES_PER_EPOCH_FOR_TEST / (BATCH_SIZE * SEQUENCE_LENGTH)
# TEST_ITER = (155//SEQUENCE_LENGTH)*(24 // BATCH_SIZE)

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, initializer, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(
      name,
      shape,
      initializer)
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def orthogonal_initializer(scale = 1.1):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    print('Warning -- You have opted to use the orthogonal_initializer function')
    def _initializer(shape, dtype=tf.float32, partition_info=None):
      flat_shape = (shape[0], np.prod(shape[1:]))
      a = np.random.normal(0.0, 1.0, flat_shape)
      u, _, v = np.linalg.svd(a, full_matrices=False)
      # pick the one with the correct shape
      q = u if u.shape == flat_shape else v
      q = q.reshape(shape) #this needs to be corrected to float32
      print('you have initialized one orthogonal matrix.')
      return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer

def msra_initializer(kl, dl):
    """
    kl for kernel size, dl for filter number
    """
    stddev = math.sqrt(2. / (kl**2 * dl))
    return tf.truncated_normal_initializer(stddev=stddev)

def weighted_loss(logits, labels, num_classes, head=None):
    """Calculate the loss from the logits and the labels.
    Args:
      logits: tensor, float - [batch_size, width, height, num_classes].
          Use vgg_fcn.up as logits.
      labels: Labels tensor, int32 - [batch_size, width, height, num_classes].
          The ground truth of your data.
      head: numpy array - [num_classes]
          Weighting the loss of each class
          Optional: Prioritize some classes
    Returns:
      loss: Loss tensor of type float.
    """
    with tf.name_scope('loss_1'):

        logits = tf.reshape(logits, (-1, num_classes))

        epsilon = tf.constant(value=1e-10)

        logits = logits + epsilon

        # consturct one-hot label array
        label_flat = tf.reshape(labels, (-1, 1))

        # should be [batch ,num_classes]
        labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))

        softmax = tf.nn.softmax(logits)

        cross_entropy = -tf.reduce_sum(tf.mul(labels * tf.log(softmax + epsilon), head), reduction_indices=[1])

        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return loss

def cal_loss(logits, labels):
    """
    loss_weight = np.asarray([
      0.0616944799702,
      3.89114328416,
      0.718496198987,
      3.24645148591,
      1.64418466389,
      0.0182122198045
    ]) # class 0~5
    """
    loss_weight = np.array([
      0.2595,
      0.1826,
      4.5640,
      0.1417,
      0.9051,
      0.3826,
      9.6446,
      1.8418,
      0.6823,
      6.2478,
      7.3614,
    ]) # class 0~10

    labels = tf.cast(labels, tf.int32)
    # return loss(logits, labels)
    return weighted_loss(logits, labels, num_classes=NUM_CLASSES, head=loss_weight)

def conv_layer_with_bn(inputT, shape, train_phase, activation=True, name=None, reuse=False):
    in_channel = shape[2]
    out_channel = shape[3]
    k_size = shape[0]
    with tf.variable_scope(name, reuse=reuse) as scope:
      kernel = _variable_with_weight_decay('weights',
                                           shape=shape,
                                           initializer=msra_initializer(k_size, out_channel),
                                           wd=0.0005)
      conv = tf.nn.conv2d(inputT, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [out_channel], tf.constant_initializer(0.0))
      bias = tf.nn.bias_add(conv, biases)
      if activation is True:
        conv_out = tf.nn.relu(batch_norm_layer(bias, train_phase, scope.name))
      else:
        conv_out = batch_norm_layer(bias, train_phase, scope.name)
    return conv_out

def get_deconv_filter(f_shape):
  """
    reference: https://github.com/MarvinTeichmann/tensorflow-fcn
  """
  width = f_shape[0]
  heigh = f_shape[0]
  f = ceil(width/2.0)
  c = (2 * f - 1 - f % 2) / (2.0 * f)
  bilinear = np.zeros([f_shape[0], f_shape[1]])
  for x in range(width):
      for y in range(heigh):
          value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
          bilinear[x, y] = value
  weights = np.zeros(f_shape)
  for i in range(f_shape[2]):
      weights[:, :, i, i] = bilinear

  init = tf.constant_initializer(value=weights,
                                 dtype=tf.float32)
  return tf.get_variable(name="up_filter", initializer=init,
                         shape=weights.shape)

def deconv_layer(inputT, f_shape, output_shape, stride=2, name=None, reuse=False):
  # output_shape = [b, w, h, c]
  strides = [1, stride, stride, 1]
  with tf.variable_scope(name, reuse=reuse):
    weights = get_deconv_filter(f_shape)
    deconv = tf.nn.conv2d_transpose(inputT, weights, output_shape,
                                        strides=strides, padding='SAME')
  return deconv

def batch_norm_layer(inputT, is_training, scope):
  return tf.cond(is_training,
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=True,
                           center=False, updates_collections=None, scope=scope+"_bn"),
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=False,
                           updates_collections=None, center=False, scope=scope+"_bn", reuse = True))

def encoder(images, phase_train, reuse=False):
  batch_size = BATCH_SIZE
  # norm1
  norm1 = tf.nn.local_response_normalization(images, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='norm1')
  # conv1
  conv1 = conv_layer_with_bn(norm1, [7, 7, IMAGE_DEPTH, 64], phase_train, name="conv1", reuse=reuse)
  # pool1
  pool1, pool1_indices = tf.nn.max_pool_with_argmax(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # conv2
  conv2 = conv_layer_with_bn(pool1, [7, 7, 64, 64], phase_train, name="conv2", reuse=reuse)

  # pool2
  pool2, pool2_indices = tf.nn.max_pool_with_argmax(conv2, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')
  # conv3
  conv3 = conv_layer_with_bn(pool2, [7, 7, 64, 64], phase_train, name="conv3", reuse=reuse)

  # pool3
  pool3, pool3_indices = tf.nn.max_pool_with_argmax(conv3, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool3')
  # conv4
  conv4 = conv_layer_with_bn(pool3, [7, 7, 64, 64], phase_train, name="conv4", reuse=reuse)

  # pool4
  pool4, pool4_indices = tf.nn.max_pool_with_argmax(conv4, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool4')
  return pool4

def decoder(inputT, phase_train, batch_size, reuse=False):
  upsample4 = deconv_layer(inputT, [2, 2, 64, 64], [batch_size, 45, 60, 64], 2, "up4", reuse=reuse)
  # decode 4
  conv_decode4 = conv_layer_with_bn(upsample4, [7, 7, 64, 64], phase_train, False, name="conv_decode4", reuse=reuse)

  # upsample 3
  upsample3= deconv_layer(conv_decode4, [2, 2, 64, 64], [batch_size, 90, 120, 64], 2, "up3", reuse=reuse)
  # decode 3
  conv_decode3 = conv_layer_with_bn(upsample3, [7, 7, 64, 64], phase_train, False, name="conv_decode3", reuse=reuse)

  # upsample2
  upsample2= deconv_layer(conv_decode3, [2, 2, 64, 64], [batch_size, 180, 240, 64], 2, "up2", reuse=reuse)
  # decode 2
  conv_decode2 = conv_layer_with_bn(upsample2, [7, 7, 64, 64], phase_train, False, name="conv_decode2", reuse=reuse)

  # upsample1
  upsample1= deconv_layer(conv_decode2, [2, 2, 64, 64], [batch_size, 360, 480, 64], 2, "up1", reuse=reuse)
  # decode4
  conv_decode1 = conv_layer_with_bn(upsample1, [7, 7, 64, 64], phase_train, False, name="conv_decode1", reuse=reuse)

  # output predicted class number (6)
  with tf.variable_scope('conv_classifier', reuse=reuse) as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[1, 1, 64, NUM_CLASSES],
                                         initializer=msra_initializer(1, 64),
                                         wd=0.0005)
    conv = tf.nn.conv2d(conv_decode1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
    conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

  return conv_classifier

def convGRU_decoder(decoder_inputs, initial_state, cell, loop_function=None, scope=None):
  with variable_scope.variable_scope(scope or "convGRU_decoder"):
    state = initial_state
    outputs = []
    prev = None
    for i, inp in enumerate(decoder_inputs):
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      output, state = cell(inp, state)
      outputs.append(output)
      if loop_function is not None:
        prev = output
  return outputs, state

def seq2seq(encoder_inputs, decoder_inputs, cell, phase_train, batch_size, dtype=dtypes.float32, scope=None):
  with variable_scope.variable_scope(scope or "basic_convGRU_seq2seq"):
    _, enc_state = rnn.rnn(cell, encoder_inputs, dtype=dtype)
    return convGRU_decoder(decoder_inputs, enc_state, cell)

def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
  epsilon = tf.constant(value=1e-10)
  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with tf.name_scope("sequence_loss_by_example"):
    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
      logit = tf.reshape(logit, (-1, NUM_CLASSES))
      logit = logit + epsilon
      # consturct one-hot label array
      target = tf.reshape(target, (-1, 1))
      # should be [batch ,num_classes]
      target = tf.reshape(tf.one_hot(target, depth=NUM_CLASSES), (-1, NUM_CLASSES))
      softmax = tf.nn.softmax(logit)
      cross_entropy = -tf.reduce_sum(tf.mul(target * tf.log(softmax + epsilon), LOSS_WEIGHT), reduction_indices=[1])
      log_perp_list.append(cross_entropy * weight)
    log_perps = math_ops.add_n(log_perp_list)
    if average_across_timesteps:
      total_size = math_ops.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps

def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
    with tf.name_scope('sequence_loss'):
        cost = tf.reduce_mean(sequence_loss_by_example(
        logits, targets, weights,
        average_across_timesteps=average_across_timesteps,
        softmax_loss_function=softmax_loss_function))
        tf.add_to_collection('losses', cost)
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss

def inference(images, labels, phase_train):
  # images [b, T, w, h, c]
  epsilon = tf.constant(value=1e-10)
  batch_size = BATCH_SIZE
  sequence_length = SEQUENCE_LENGTH
  # first feed first t images for one batch
  # with tf.variable_scope("inputs") as scope:
  images = tf.transpose(images, [1,0,2,3,4])
  labels = tf.transpose(labels, [1,0,2,3,4])
    # become [T , b, w, h ,c]
  images = tf.unpack(images)
  labels = tf.unpack(labels)

    # get encoder input as list
  encoder_inputs1 = []

  for i in range(sequence_length):
    if i > 0:
      encoder_inputs1.append(encoder(images[i], phase_train, reuse=True))
    else:
      encoder_inputs1.append(encoder(images[i], phase_train))

  cell = convLSTM.ConvLSTMCell(64, k_size=3, height=23, width=30, initializer=orthogonal_initializer())
  cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=0.5, output_keep_prob=0.5)
  stacked_lstm = convLSTM.MultiRNNCell([cell] * 2)
  # output, state = seq2seq(encoder_inputs, encoder_inputs, cell, phase_train, batch_size)
  with tf.variable_scope("basic_convGRU_seq2seq"):
    output, enc_state = rnn.rnn(stacked_lstm, encoder_inputs1, dtype=tf.float32)

  decoder_outputs1 = []
  for i in range(sequence_length):
    if i > 0:
      decoder_outputs1.append(decoder(output[i], phase_train, batch_size, reuse=True))
    else:
      decoder_outputs1.append(decoder(output[i], phase_train, batch_size))
  # decoder_outputs = decoder_outputs1[0]
  # print("decoders", decoder_outputs)
  weights = [tf.constant(1.0, dtype=tf.float32) for i in range(sequence_length)]
  loss = sequence_loss(decoder_outputs1, labels, weights)
  logit = decoder_outputs1
  # loss = cal_loss(logit, labels[0])

  return loss, logit

def train(total_loss, global_step):
    batch_size = BATCH_SIZE
    total_sample = 274
    num_batches_per_epoch = 274/1
    """ fix lr """
    lr = INITIAL_LEARNING_RATE
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
      # opt = tf.train.GradientDescentOptimizer(lr)
      # opt = tf.train.MomentumOptimizer(lr, 0.9)
      opt = tf.train.AdamOptimizer(lr)
      grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        tf.histogram_summary(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
      train_op = tf.no_op(name='train')

    return train_op

def seq_test():
  # testing should set BATCH_SIZE = 1
  batch_size = 1
  seq_length = 3
  image_filenames, label_filenames = get_filename_list("/tmp3/first350/SegNet-Tutorial/CamVid/test.txt")

  test_data_node = tf.placeholder(
        tf.float32,
        shape=[batch_size, seq_length, 360, 480, 3])

  test_labels_node = tf.placeholder(tf.int64, shape=[batch_size, seq_length, 360, 480, 1])

  phase_train = tf.placeholder(tf.bool, name='phase_train')

  loss, logits = inference(test_data_node, test_labels_node, phase_train)

  # pred = tf.argmax(logits, dimension=3)

  # get moving avg
  variable_averages = tf.train.ExponentialMovingAverage(
                      MOVING_AVERAGE_DECAY)
  variables_to_restore = variable_averages.variables_to_restore()

  saver = tf.train.Saver(variables_to_restore)

  with tf.Session() as sess:
    # Load checkpoint
    saver.restore(sess, "/tmp3/first350/TensorFlow/Logs_seq/seq_model.ckpt-2000" )
    images, labels = get_all_test_data_seq(image_filenames, label_filenames, seq_length)
    # threads = tf.train.start_queue_runners(sess=sess)
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for image_batch, label_batch  in zip(images, labels):
      print(image_batch.shape, label_batch.shape)
      feed_dict = {
        test_data_node: image_batch,
        test_labels_node: label_batch,
        phase_train: False
      }
      dense_prediction = sess.run(logits, feed_dict=feed_dict)
      print(dense_prediction[0].shape)
      hist += Utils.get_hist_seq(dense_prediction, label_batch, batch_size, seq_length, NUM_CLASSES)
    acc_total = np.diag(hist).sum() / hist.sum()
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print("acc: ", acc_total)
    print("mean IU: ", np.nanmean(iu))

def seq_main():
  image_filenames, label_filenames = get_filename_list_seq("/tmp3/first350/SegNet-Tutorial/CamVid/train.txt", SEQUENCE_LENGTH)
  val_image_filenames, val_label_filenames = get_filename_list_seq("/tmp3/first350/SegNet-Tutorial/CamVid/test.txt", SEQUENCE_LENGTH)
  batch_size = BATCH_SIZE
  seq_length = SEQUENCE_LENGTH
  max_steps = 4000
  train_dir = "/tmp3/first350/TensorFlow/Logs_seq"
  # ff = get_miccai_filename(seq_length)
  with tf.Graph().as_default():
    train_data_node = tf.placeholder(
          tf.float32,
          shape=[batch_size, seq_length, 360, 480, 3])

    train_labels_node = tf.placeholder(tf.int64, shape=[batch_size, seq_length, 360, 480, 1])

    phase_train = tf.placeholder(tf.bool, name='phase_train')

    global_step = tf.Variable(0, trainable=False)

    # im_seq, la_seq = MiccaiInputs_seq(ff, batch_size, seq_length)

    image_seq, label_seq = CamVidInputs_seq(image_filenames, label_filenames, batch_size, seq_length=seq_length)

    val_images, val_labels = CamVidInputs_seq(val_image_filenames, val_label_filenames, batch_size, seq_length=seq_length)

    loss, eval_prediction = inference(train_data_node, train_labels_node, phase_train)

    train_op = train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=50)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    with tf.Session() as sess:
      # Build an initialization operation to run below.
      init = tf.initialize_all_variables()

      # Start running operations on the Graph.
      sess.run(init)

      threads = tf.train.start_queue_runners(sess=sess)

      summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)
      average_pl = tf.placeholder(tf.float32)
      acc_pl = tf.placeholder(tf.float32)
      iu_pl = tf.placeholder(tf.float32)
      average_summary = tf.scalar_summary("test_average_loss", average_pl)
      acc_summary = tf.scalar_summary("test_accuracy", acc_pl)
      iu_summary = tf.scalar_summary("Mean_IU", iu_pl)
      for step in range(max_steps):
        image_batch ,label_batch = sess.run([image_seq, label_seq])
        # image_batch, label_batch = get_miccai_data(ff[0:250], batch_size, seq_length)
        # freq = Utils.count_freq(label_batch, batch_size)
        feed_dict = {
           train_data_node :image_batch,
           train_labels_node: label_batch,
           phase_train: True
        }
        start_time = time.time()
        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
        duration = time.time() - start_time
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
        if step % 20 == 0:
          num_examples_per_step = batch_size
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
          print (format_str % (datetime.now(), step, loss_value,
                               examples_per_sec, sec_per_batch))
          pred = sess.run(eval_prediction, feed_dict=feed_dict)
          Utils.eval_seq(pred, label_batch, batch_size, seq_length, NUM_CLASSES)
          # freq = Utils.count_freq(label_batch, batch_size)

        if step % 100 == 0:
          print("start testing.....")
          total_val_loss = 0.0
          hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
          brain_index = 0
          depth_index = 0
          for test_step in range(TEST_ITER):
            val_images_batch, val_labels_batch = sess.run([val_images, val_labels])
            """
            val_images_batch, val_labels_batch = get_miccai_test_data(ff[250: 274], batch_size, seq_length, brain_index, depth_index)
            depth_index += seq_length
            if (depth_index + seq_length > 155):
              brain_index += batch_size
              depth_index = 0
            """

            _val_loss, _val_pred = sess.run([loss, eval_prediction], feed_dict={
              train_data_node: val_images_batch,
              train_labels_node: val_labels_batch,
              phase_train: True
            })
            total_val_loss += _val_loss
            hist += Utils.get_hist_seq(_val_pred, val_labels_batch, batch_size, seq_length, NUM_CLASSES)

          print("val loss: ", total_val_loss / TEST_ITER)
          acc_total = np.diag(hist).sum() / hist.sum()
          iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
          test_summary_str = sess.run(average_summary, feed_dict={average_pl: total_val_loss / TEST_ITER})
          acc_summary_str = sess.run(acc_summary, feed_dict={acc_pl: acc_total})
          iu_summary_str = sess.run(iu_summary, feed_dict={iu_pl: np.nanmean(iu)})
          Utils.print_hist_summery(hist, NUM_CLASSES)
          summary_str = sess.run(summary_op, feed_dict=feed_dict)
          summary_writer.add_summary(summary_str, step)
          summary_writer.add_summary(test_summary_str, step)
          summary_writer.add_summary(acc_summary_str, step)
          summary_writer.add_summary(iu_summary_str, step)
        # Save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == max_steps:
          checkpoint_path = os.path.join(train_dir, 'seq_model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)



if __name__ == "__main__":
  seq_main()
  # seq_test()
