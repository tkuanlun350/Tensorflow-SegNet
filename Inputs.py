import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import os, sys
import numpy as np
import math
import Image
import skimage
import skimage.io
import SimpleITK as sitk

IMAGE_HEIGHT = 360
IMAGE_WIDTH = 480
IMAGE_DEPTH = 3

NUM_CLASSES = 11
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 367
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 101
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 3-D Tensor of [height, width, 1] type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 3D tensor of [batch_size, height, width ,1] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 1
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  # tf.image_summary('images', images)

  return images, label_batch

def CamVid_reader_seq(filename_queue, seq_length):
  image_seq_filenames = tf.split(0, seq_length, filename_queue[0])
  label_seq_filenames = tf.split(0, seq_length, filename_queue[1])

  image_seq = []
  label_seq = []
  for im ,la in zip(image_seq_filenames, label_seq_filenames):
    imageValue = tf.read_file(tf.squeeze(im))
    labelValue = tf.read_file(tf.squeeze(la))
    image_bytes = tf.image.decode_png(imageValue)
    label_bytes = tf.image.decode_png(labelValue)
    image = tf.cast(tf.reshape(image_bytes, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH)), tf.float32)
    label = tf.cast(tf.reshape(label_bytes, (IMAGE_HEIGHT, IMAGE_WIDTH, 1)), tf.int64)
    image_seq.append(image)
    label_seq.append(label)
  return image_seq, label_seq

def CamVid_reader(filename_queue):

  image_filename = filename_queue[0]
  label_filename = filename_queue[1]

  imageValue = tf.read_file(image_filename)
  labelValue = tf.read_file(label_filename)

  image_bytes = tf.image.decode_png(imageValue)
  label_bytes = tf.image.decode_png(labelValue)

  image = tf.reshape(image_bytes, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
  label = tf.reshape(label_bytes, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))

  return image, label

def Miccai_reader_seq(filename_queue, seq_length):
  im_seq = []
  la_seq = []
  seed_filename = np.random.choice(filename_queue, 1)
  print(seed_filename)
  seed = np.random.rand() % (155 - seq_length)
  seed=100
  for i in seed_filename:
      im = [0,0,0,0] # to reserve order
      for ii in os.listdir(i):
          fpath = i + "/" + ii
          for j in os.listdir(fpath):
            if ".mha" not in j or "N4" in j:
              continue
            if "OT" in j:
              la = sitk.GetArrayFromImage(sitk.ReadImage(fpath + "/" + j))[seed : seed+seq_length]
            elif "Flair" in j:
              im[0] = sitk.GetArrayFromImage(sitk.ReadImage(fpath + "/" + j))[seed : seed+seq_length]
            elif "T1c" in j:
              im[1] = sitk.GetArrayFromImage(sitk.ReadImage(fpath + "/" + j))[seed : seed+seq_length]
            elif "T1" in j:
              im[2] = sitk.GetArrayFromImage(sitk.ReadImage(fpath + "/" + j))[seed : seed+seq_length]
            elif "T2" in j:
              im[3] = sitk.GetArrayFromImage(sitk.ReadImage(fpath + "/" + j))[seed : seed+seq_length]
      im_seq = tf.constant(np.transpose(im,[1,2,3,0]), tf.float32)
      la_seq = tf.constant(la, tf.uint8, shape=[seq_length,240,240,1])
  return im_seq, la_seq


def get_filename_list_seq(path, seq_length):
  fd = open(path)
  total_train_file = 367
  im_seq = []
  label_seq = []
  image_filenames = []
  label_filenames = []
  filenames = []
  index = 0
  for i in fd:
    if index%seq_length == 0 and index != 0:
      image_filenames.append(im_seq)
      label_filenames.append(label_seq)
      im_seq = []
      label_seq = []

    i = i.strip().split(" ")
    im_seq.append(i[0])
    label_seq.append(i[1])
    index += 1
  print("im, label seq: ", len(image_filenames), len(label_filenames))
  return image_filenames, label_filenames

def get_filename_list(path):
  fd = open(path)
  image_filenames = []
  label_filenames = []
  filenames = []
  for i in fd:
    i = i.strip().split(" ")
    image_filenames.append(i[0])
    label_filenames.append(i[1])
  return image_filenames, label_filenames

def MiccaiInputs_seq(filenames, batch_size, seq_length):
  image_seq, label_seq = Miccai_reader_seq(filenames, seq_length)
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(73 *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CamVid seq_images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(image_seq, label_seq,
                                         min_queue_examples, batch_size,
                                         shuffle=True)

def CamVidInputs_seq(image_filenames, label_filenames, batch_size, seq_length):
  images = ops.convert_to_tensor(image_filenames, dtype=dtypes.string)
  labels = ops.convert_to_tensor(label_filenames, dtype=dtypes.string)
  print("seq im, la filenames", images.get_shape(), labels.get_shape())
  filename_queue = tf.train.slice_input_producer([images, labels], shuffle=True)
  image_seq, label_seq = CamVid_reader_seq(filename_queue, seq_length)
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(73 *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CamVid seq_images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(image_seq, label_seq,
                                         min_queue_examples, batch_size,
                                         shuffle=True)

def CamVidInputs(image_filenames, label_filenames, batch_size):

  images = ops.convert_to_tensor(image_filenames, dtype=dtypes.string)
  labels = ops.convert_to_tensor(label_filenames, dtype=dtypes.string)

  filename_queue = tf.train.slice_input_producer([images, labels], shuffle=True)

  image, label = CamVid_reader(filename_queue)
  reshaped_image = tf.cast(image, tf.float32)

  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CamVid images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(reshaped_image, label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)

def get_all_test_data(im_list, la_list):
  images = []
  labels = []
  index = 0
  for im_filename, la_filename in zip(im_list, la_list):
    im = np.array(skimage.io.imread(im_filename), np.float32)
    im = im[np.newaxis]
    la = skimage.io.imread(la_filename)
    la = la[np.newaxis]
    la = la[...,np.newaxis]
    images.append(im)
    labels.append(la)
  return images, labels

def get_all_test_data_seq(im_list, la_list, seq_length):
  images = []
  labels = []
  im1 = []
  la1 = []
  index = 0
  for im_filename, la_filename in zip(im_list, la_list):
    if index % 3 == 0 and index != 0:
        im1.append(np.array(images)[np.newaxis])
        la1.append(np.array(labels)[np.newaxis])
        images = []
        labels = []
    im = np.array(skimage.io.imread(im_filename), np.float32)
    #im = im[np.newaxis]
    la = skimage.io.imread(la_filename)
    #la = la[np.newaxis]
    la = la[...,np.newaxis]
    images.append(im)
    labels.append(la)
    index += 1
  return im1, la1

def get_miccai_filename(seq_length):
  path="/tmp2/r04921120/MIRA/BRATS2015_Training-2/HGG/"
  path2="/tmp2/r04921120/MIRA/BRATS2015_Training-2/LGG/"
  dl = os.listdir(path)
  filenames = []
  labels = []
  files = []
  for index,p in enumerate(dl):
      filenames.append(path+"/"+p)
  for index,p in enumerate(os.listdir(path2)):
      filenames.append(path2+"/"+p)
  print("filename len: ", len(filenames))
  np.random.shuffle(filenames)
  return filenames
def get_miccai_data(filenames, batch_size, seq_length):
  im_seq = []
  la_seq = []
  sample_len = 1
  seed_filename = np.random.choice(filenames, batch_size)
  for i in seed_filename:
      im = [0,0,0,0] # to reserve order
      seed = np.random.rand() % (155 - sample_len*seq_length - 50 ) + 50
      for ii in os.listdir(i):
          fpath = i + "/" + ii
          for j in os.listdir(fpath):
            if ".mha" not in j or "N4" in j:
              continue
            if "OT" in j:
              la = sitk.GetArrayFromImage(sitk.ReadImage(fpath + "/" + j))[seed : seed+seq_length*sample_len : sample_len]
            elif "Flair" in j:
              im[0] = sitk.GetArrayFromImage(sitk.ReadImage(fpath + "/" + j))[seed : seed+seq_length*sample_len : sample_len]
            elif "T1c" in j:
              im[1] = sitk.GetArrayFromImage(sitk.ReadImage(fpath + "/" + j))[seed : seed+seq_length*sample_len : sample_len]
            elif "T1" in j:
              im[2] = sitk.GetArrayFromImage(sitk.ReadImage(fpath + "/" + j))[seed : seed+seq_length*sample_len : sample_len]
            elif "T2" in j:
              im[3] = sitk.GetArrayFromImage(sitk.ReadImage(fpath + "/" + j))[seed : seed+seq_length*sample_len : sample_len]
      im_seq.append(np.transpose(im,[1,2,3,0]))
      la_seq.append(la[..., np.newaxis])
  return np.asarray(im_seq), np.asarray(la_seq)

def get_miccai_test_data(filenames, batch_size, seq_length, brain_index, depth_index):
  im_seq = []
  la_seq = []
  sample_len = 1
  seed_filename = filenames[brain_index: brain_index+batch_size]
  for i in seed_filename:
      im = [0,0,0,0] # to reserve order
      seed = depth_index
      for ii in os.listdir(i):
          fpath = i + "/" + ii
          for j in os.listdir(fpath):
            if ".mha" not in j or "N4" in j:
              continue
            if "OT" in j:
              la = sitk.GetArrayFromImage(sitk.ReadImage(fpath + "/" + j))[seed : seed+seq_length*sample_len : sample_len]
            elif "Flair" in j:
              im[0] = sitk.GetArrayFromImage(sitk.ReadImage(fpath + "/" + j))[seed : seed+seq_length*sample_len : sample_len]
            elif "T1c" in j:
              im[1] = sitk.GetArrayFromImage(sitk.ReadImage(fpath + "/" + j))[seed : seed+seq_length*sample_len : sample_len]
            elif "T1" in j:
              im[2] = sitk.GetArrayFromImage(sitk.ReadImage(fpath + "/" + j))[seed : seed+seq_length*sample_len : sample_len]
            elif "T2" in j:
              im[3] = sitk.GetArrayFromImage(sitk.ReadImage(fpath + "/" + j))[seed : seed+seq_length*sample_len : sample_len]
      im_seq.append(np.transpose(im,[1,2,3,0]))
      la_seq.append(la[..., np.newaxis])
  return np.asarray(im_seq), np.asarray(la_seq)
