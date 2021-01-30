#!/usr/bin/python
#
# Copyright 2020 Brown Visual Computing Lab / Authors of the accompanying paper Matryodshka #
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by Brown Visual Computing Lab / Authors of the accompanying paper Matryodshka

"""Network definitions for MSI prediction networks.
"""
from __future__ import division
import numpy as np
import tensorflow as tf
import sys
from tensorflow.contrib import slim

flags = tf.app.flags
FLAGS = flags.FLAGS

### PATCH SQUARED DIFFERENCE ###
if False:
  from tensorflow.python.ops import math_ops
  def squared_difference(*args, **kargs):
      x, y  = args[:2]
      return tf.square(x - y)

  math_ops.__dict__['squared_difference'] = squared_difference
### PATCH SQUARED DIFFERENCE ###

### SLIM REPLACEMENT OPS ###
def get_pool_sizes(height, width):
  curr_height = height
  curr_width = width
  height_pools = []
  width_pools = []

  while curr_height > 1 or curr_width > 1:
    # Reduce height
    reduced_height = False

    for i in range(14, 1, -1):
      if (curr_height % i) == 0:
        height_pools.append(i)
        curr_height = curr_height / i
        reduced_height = True
        break

    if not reduced_height:
      height_pools.append(1)

    # Reduce width
    reduced_width = False

    for i in range(14, 1, -1):
      if (curr_width % i) == 0:
        width_pools.append(i)
        curr_width = curr_width / i
        reduced_width = True
        break

    if not reduced_width:
      width_pools.append(1)

    if not reduced_height and not reduced_width:
      break

  return [[1, h, w, 1] for h, w in zip(height_pools, width_pools)]

def reduce_mean(input, inputs_shape, norm_axes):
  mean = input

  # Perform pooling
  pools = get_pool_sizes(inputs_shape[1], inputs_shape[2])

  for pool in pools:
    mean = tf.nn.avg_pool(mean, pool, pool, padding='VALID')

  # Reduce
  return tf.reduce_mean(mean, norm_axes, keepdims=True)

def layer_norm(input, scope):

  inputs_shape = input.get_shape().as_list()
  params_shape = inputs_shape[-1:]
  with tf.name_scope(scope):
    # Scale, offset, eps
    beta = tf.Variable(
      tf.zeros(params_shape, dtype=tf.float32),
      name='beta'
    )

    gamma = tf.Variable(
      tf.zeros(params_shape, dtype=tf.float32),
      name='gamma'
    )

    norm_axes = list(range(1, len(inputs_shape)))
    mean = reduce_mean(input, inputs_shape, norm_axes)
    var = reduce_mean(tf.square(input - mean), inputs_shape, norm_axes)
    output = gamma * (input - mean) / tf.sqrt(var) + beta

    return tf.reshape(output, inputs_shape)

def conv2d(input, channels, kernel, scope=None, stride=1, rate=1, padding=None, slice=None, activation_fn=tf.nn.relu, normalizer_fn=layer_norm):

  with tf.name_scope(scope) as my_scope:
    if padding != None:
      input = tf.pad(input, [[0, 0], padding[:2], padding[2:], [0, 0]])
    conv = conv2d_helper(input, channels, kernel, scope=scope, stride=stride, rate=rate)
    if slice != None:
      conv = conv[:, slice[0]:slice[1], slice[2]:slice[3], :]

    if normalizer_fn != None:
      if normalizer_fn == slim.layer_norm:
        conv = normalizer_fn(conv, scope=scope + '/LayerNorm')
      else:
        conv = normalizer_fn(conv, scope='LayerNorm')

    if activation_fn != None:
      conv = activation_fn(conv, name=scope)

  return conv

def conv2d_transpose(input, channels, kernel, scope=None, stride=1, padding=None, slice=None, activation_fn=tf.nn.relu, normalizer_fn=layer_norm):

  with tf.name_scope(scope) as my_scope:
    if FLAGS.smoothed:
      conv = conv2d_transpose_helper_smoothed(input, channels, kernel, scope=scope, stride=stride, padding=padding)
    else:
      conv = conv2d_transpose_helper(input, channels, kernel, scope=scope, stride=stride, padding=padding)

    if slice != None:
      conv = conv[:, slice[0]:slice[1], slice[2]:slice[3], :]

    if normalizer_fn != None:
      if normalizer_fn == slim.layer_norm:
        conv = normalizer_fn(conv, scope=scope + '/LayerNorm')
      else:
        conv = normalizer_fn(conv, scope='LayerNorm')

    if activation_fn != None:
      conv = activation_fn(conv, name=scope)

  return conv

def separable_conv2d(input, kernel, scope=None, stride=1, rate=1, depth_multiplier=1, padding=None, slice=None):

  with tf.name_scope(scope) as my_scope:

    if padding != None:
        input = tf.pad(input, [[0, 0], padding[:2], padding[2:], [0, 0]])

    conv = separable_conv2d_helper(input, kernel, scope=scope, stride=stride, rate=rate, depth_multiplier=depth_multiplier)

    if slice != None:
      conv = conv[:, slice[0]:slice[1], slice[2]:slice[3], :]

  return conv

def conv2d_helper(input, channels, kernel, scope=None, stride=1, rate=1):
  kernel = tf.Variable(tf.truncated_normal([kernel[0], kernel[1], tf.shape(input)[-1], channels], dtype=tf.float32,
                                          stddev=1e-1), name='weights')

  conv = tf.nn.conv2d(input, kernel, [1, stride, stride, 1], padding='VALID', dilations=[1, rate, rate, 1])

  return conv

def separable_conv2d_helper(input, kernel, scope=None, stride=1, rate=1, depth_multiplier=1):
  kernel = tf.Variable(tf.truncated_normal([kernel[0], kernel[1], tf.shape(input)[-1], depth_multiplier], dtype=tf.float32,
                                          stddev=1e-1), name='depthwise_weights')

  conv = tf.nn.depthwise_conv2d(input, kernel, [1, stride, stride, 1], padding='VALID', dilations=[rate, rate])
  return conv

def conv2d_transpose_helper_smoothed(input, channels, kernel, scope=None, stride=1, padding=None):
  batch, height, width, _ = input.get_shape().as_list()

  if FLAGS.smoothed:
      kernel = tf.Variable(tf.truncated_normal([kernel[0], kernel[1],tf.shape(input)[-1], channels],
                                               dtype=tf.float32, stddev=1e-1), name='weights')
  else:
      kernel = tf.Variable(tf.truncated_normal([kernel[0], kernel[1], channels, tf.shape(input)[-1]], dtype=tf.float32,
                                                stddev=1e-1), name='weights')
      kernel = tf.transpose(kernel, [0, 1, 3, 2])

  input = tf.image.resize_images(input, [stride * height, stride * width], align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  if padding != None:
    input = tf.pad(input, [[0, 0], padding[:2], padding[2:], [0, 0]])

  conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='VALID')
  return conv

def conv2d_transpose_helper(input, channels, kernel, scope=None, stride=1, padding=None):
  batch, height, width, _ = input.get_shape().as_list()
  kernel = tf.Variable(tf.truncated_normal([kernel[0], kernel[1], channels, tf.shape(input)[-1]], dtype=tf.float32,
                                          stddev=1e-1), name='weights')
  conv = tf.nn.conv2d_transpose(input, kernel, strides=[1, stride, stride, 1], output_shape=[batch, height * stride + 2, width * stride + 2, channels], padding='VALID')
  return conv

def depthwise_separable_conv2D(inputs,num_pwc_filters,scope=None,width_multiplier=1,stride=1,rate=1,normalizer_fn=layer_norm,padding=None,slice=None):
    """ Helper function to build the depth-wise separable convolution layer.
    """
    # num_pwc_filters = round(num_pwc_filters * width_multiplier)
    _stride = stride
    sc = str(scope)
    # skip pointwise by setting num_outputs=None
    depthwise_conv = slim.separable_convolution2d(inputs,
                                                  num_outputs=None,
                                                  stride=_stride,
                                                  depth_multiplier=1,
                                                  kernel_size=[3, 3],
                                                  rate=rate,
                                                  scope=sc+'/depthwise_conv')
    # depthwise_conv = separable_conv2d(inputs, stride=_stride, depth_multiplier=width_multiplier, kernel=[3, 3], rate=rate, scope=sc+'/depthwise_conv',padding=padding, slice=slice)
    ln = normalizer_fn(depthwise_conv, scope=sc+'/dw_layer_norm')
    pointwise_conv = conv2d(ln, num_pwc_filters, [1, 1], scope=sc+'/pointwise_conv', rate = 2, normalizer_fn=None)
    ln = normalizer_fn(pointwise_conv, scope=sc+'/pw_layer_norm')
    return ln

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size,
            align_corners=True)
### SLIM REPLACEMENT OPS ###

def add_coords(input):
    #based on code from spherical view synthesis 3dv paper
    _, height, width, _ = input.get_shape().as_list()
    xx_ones = tf.ones([1, height], dtype=tf.float32)
    xx_ones = tf.expand_dims(xx_ones, -1)
    xx_range = tf.expand_dims(tf.range(width, dtype=tf.float32),0)
    xx_range = tf.expand_dims(xx_range, 1)
    xx_channel = tf.matmul(xx_ones, xx_range)
    xx_channel = tf.expand_dims(xx_channel, -1)
    yy_ones = tf.ones([1, width], dtype=tf.float32)
    yy_ones = tf.expand_dims(yy_ones, 1)
    yy_range = tf.expand_dims(tf.range(height, dtype=tf.float32),0)
    yy_range = tf.expand_dims(yy_range, -1)
    yy_channel = tf.matmul(yy_range, yy_ones)
    yy_channel = tf.expand_dims(yy_channel, -1)
    xx_channel = xx_channel / (width - 1)
    yy_channel = yy_channel / (height - 1)
    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1
    return tf.concat([input, xx_channel, yy_channel], axis=-1)

def add_sph_coords(input):
    _, height, width, _ = input.get_shape().as_list()
    coord = np.tile(np.expand_dims(np.linspace(-np.pi / 2.0, np.pi / 2.0, height), axis=-1), [1, width])
    coord = np.abs(np.sin(coord))
    coord = tf.convert_to_tensor(np.expand_dims(np.expand_dims(coord, axis=2), axis=0), tf.float32)  + input[:, :, :, :1] / sys.float_info.max
    return tf.concat([input, coord], axis=3)

def coord_conv2d(input, ngf, kernel, scope=None, stride=1, rate=1, activation_fn=tf.nn.relu, normalizer_fn=slim.layer_norm, add_coords=True):
    if add_coords:
        input = add_sph_coords(input)
    return slim.conv2d(input, ngf, kernel, scope=scope, stride=stride, rate=rate, activation_fn=tf.nn.relu, normalizer_fn=slim.layer_norm)

def coord_inf_conv2d(input, channels, kernel, scope=None, stride=1, rate=1, padding=None, slice=None, activation_fn=tf.nn.relu, normalizer_fn=layer_norm, add_coords=True):
    if add_coords:
        input = add_sph_coords(input)
    return conv2d(input, channels, kernel, scope=scope, stride=stride, rate=rate, padding=padding, slice=slice, activation_fn=activation_fn, normalizer_fn=normalizer_fn)

def coord_transpose_conv2d(input, ngf, kernel, scope=None, stride=1, rate=1):
    input = add_coords(input)
    return slim.conv2d_transpose(input, ngf, kernel, scope=scope, stride=stride)

def to_placeholder(tensor, name=None):
    if not isinstance(tensor, tf.Tensor):
        tensor = tf.convert_to_tensor(tensor)
    if name == None:
        name = 'inp_' + tensor.op.name
    return tf.placeholder(tensor.dtype, tensor.shape, name=name)

def wrap_pad(inputs, left_pad, right_pad):
    left = inputs[:,:,-left_pad:,:]
    left = tf.reshape(left, [tf.shape(inputs)[0], tf.shape(inputs)[1], left_pad, tf.shape(inputs)[3]])
    right = inputs[:,:,:right_pad,:]
    right = tf.reshape(right, [tf.shape(inputs)[0], tf.shape(inputs)[1], right_pad, tf.shape(inputs)[3]])
    padded_inputs = tf.concat([left, inputs, right], axis=-2)
    padded_inputs = tf.pad(padded_inputs, [[0,0],[left_pad,right_pad],[0,0],[0,0]], mode="CONSTANT")
    return padded_inputs

def msi_inference_net(inputs, num_outputs, ngf=64, vscope='net', reuse_weights=False):
  """Network definition for multiplane image (msi) inference.

  Args:
    inputs: stack of input images [batch, height, width, input_channels]
    num_outputs: number of output channels
    ngf: number of features for the first conv layer
    vscope: variable scope
    reuse_weights: whether to reuse weights (for weight sharing)
  Returns:
    pred: network output at the same spatial resolution as the inputs.
  """
  if FLAGS.net_only:
    inputs = to_placeholder(inputs, name='plane_sweep_input')
    df = 'NHWC'
    pd = 'VALID'
  else:
    df = 'NHWC'
    pd = 'VALID'

  with tf.variable_scope(vscope, reuse=reuse_weights):
    with slim.arg_scope(
        [slim.conv2d, slim.conv2d_transpose], normalizer_fn=slim.layer_norm, padding=pd, data_format=df):
      # Encoder
      cnv1_1 = conv2d(inputs, ngf, [3, 3], scope='conv1_1', stride=1, padding=[1, 1, 1, 1])
      cnv1_2 = conv2d(cnv1_1, ngf * 2, [3, 3], scope='conv1_2', stride=2, padding=[1, 1, 1, 1])

      cnv2_1 = conv2d(cnv1_2, ngf * 2, [3, 3], scope='conv2_1', stride=1, padding=[1, 1, 1, 1])
      cnv2_2 = conv2d(cnv2_1, ngf * 4, [3, 3], scope='conv2_2', stride=2, padding=[1, 1, 1, 1])

      cnv3_1 = conv2d(cnv2_2, ngf * 4, [3, 3], scope='conv3_1', stride=1, padding=[1, 1, 1, 1])
      cnv3_2 = conv2d(cnv3_1, ngf * 4, [3, 3], scope='conv3_2', stride=1, padding=[1, 1, 1, 1])
      cnv3_3 = conv2d(cnv3_2, ngf * 8, [3, 3], scope='conv3_3', stride=2, padding=[1, 1, 1, 1])

      cnv4_1 = conv2d(cnv3_3, ngf * 8, [3, 3], scope='conv4_1', stride=1, rate=2, padding=[2, 3, 2, 3], slice=[0, int(FLAGS.height / 8), 0, int(FLAGS.width / 8)])
      cnv4_2 = conv2d(cnv4_1, ngf * 8, [3, 3], scope='conv4_2', stride=1, rate=2, padding=[2, 3, 2, 3], slice=[0, int(FLAGS.height / 8), 0, int(FLAGS.width / 8)])
      cnv4_3 = conv2d(cnv4_2, ngf * 8, [3, 3], scope='conv4_3', stride=1, rate=2, padding=[2, 3, 2, 3], slice=[0, int(FLAGS.height / 8), 0, int(FLAGS.width / 8)])

      # Decoder
      skip = tf.concat([cnv4_3, cnv3_3], axis=3)
      if FLAGS.smoothed:
        cnv6_1 = conv2d_transpose(skip, ngf * 4, [4, 4], scope='conv6_1', stride=2, padding=[1, 2, 1, 2])
      else:
        cnv6_1 = conv2d_transpose(skip, ngf * 4, [4, 4], scope='conv6_1', stride=2, slice=[2, int(FLAGS.height / 4) + 2, 2, int(FLAGS.width / 4) + 2])
      cnv6_2 = conv2d(cnv6_1, ngf * 4, [3, 3], scope='conv6_2', stride=1, padding=[1, 1, 1, 1])
      cnv6_3 = conv2d(cnv6_2, ngf * 4, [3, 3], scope='conv6_3', stride=1, padding=[1, 1, 1, 1])

      skip = tf.concat([cnv6_3, cnv2_2], axis=3)
      if FLAGS.smoothed:
        cnv7_1 = conv2d_transpose(skip, ngf * 2, [4, 4], scope='conv7_1', stride=2, padding=[1, 2, 1, 2])
      else:
        cnv7_1 = conv2d_transpose(skip, ngf * 2, [4, 4], scope='conv7_1', stride=2, slice=[2, int(FLAGS.height / 2) + 2, 2, int(FLAGS.width / 2) + 2])
      cnv7_2 = conv2d(cnv7_1, ngf * 2, [3, 3], scope='conv7_2', stride=1, padding=[1, 1, 1, 1])

      skip = tf.concat([cnv7_2, cnv1_2], axis=3)
      if FLAGS.smoothed:
        cnv8_1 = conv2d_transpose(skip, ngf, [4, 4], scope='conv8_1', stride=2, padding=[1, 2, 1, 2])
      else:
        cnv8_1 = conv2d_transpose(skip, ngf, [4, 4], scope='conv8_1', stride=2, slice=[2, int(FLAGS.height) + 2, 2, int(FLAGS.width) + 2])
      cnv8_2 = conv2d(cnv8_1, ngf, [3, 3], scope='conv8_2', stride=1, padding=[1, 1, 1, 1])

      # Output
      feat = cnv8_2
      pred = conv2d(
          feat,
          num_outputs,
          [1, 1],
          stride=1,
          activation_fn=tf.nn.tanh,
          normalizer_fn=None,
          scope='color_pred')

  msi_output = pred
  if FLAGS.net_only and FLAGS.which_color_pred == 'blend_psv':
      msi_output = tf.transpose(msi_output, [0, 3, 1, 2])
      output_h = 8
      msi_output = msi_output[:, :64, :, :]
      msi_output = tf.reshape(msi_output, [1, 8, output_h, FLAGS.height, FLAGS.width])
      msi_output = tf.transpose(msi_output, [0, 1, 3, 2, 4])
      msi_output = tf.reshape(msi_output, [1, 8 * FLAGS.height, output_h * FLAGS.width], name='msi_output')
  elif FLAGS.net_only and FLAGS.which_color_pred == 'alpha_only':
      msi_output = tf.transpose(msi_output, [0, 3, 1, 2])
      msi_output = msi_output[:, :, :, :]
      msi_output = tf.reshape(msi_output, [1, 8, 4, FLAGS.height, FLAGS.width])
      msi_output = tf.transpose(msi_output, [0, 1, 3, 2, 4])
      msi_output = tf.reshape(msi_output, [1, 8 * FLAGS.height, 4 * FLAGS.width], name='msi_output')
  else:
      msi_output = tf.identity(msi_output, name='msi_output')
  return msi_output

def msi_train_net(inputs, num_outputs, ngf=64, vscope='net', reuse_weights=False):
  """Network definition for multiplane image (msi) inference.

  Args:
    inputs: stack of input images [batch, height, width, input_channels]
    num_outputs: number of output channels
    ngf: number of features for the first conv layer
    vscope: variable scope
    reuse_weights: whether to reuse weights (for weight sharing)
  Returns:
    pred: network output at the same spatial resolution as the inputs.
  """
  with tf.variable_scope(vscope, reuse=reuse_weights):
    with slim.arg_scope(
        [slim.conv2d, slim.conv2d_transpose], normalizer_fn=slim.layer_norm):

      cnv1_1 = slim.conv2d(wrap_pad(inputs, 1, 1), ngf, [3, 3], scope='conv1_1', stride=1, padding="VALID")

      cnv1_2 = slim.conv2d(wrap_pad(cnv1_1, 1, 1), ngf * 2, [3, 3], scope='conv1_2', stride=2, padding="VALID")

      cnv2_1 = slim.conv2d(wrap_pad(cnv1_2, 1, 1), ngf * 2, [3, 3], scope='conv2_1', stride=1, padding="VALID")

      cnv2_2 = slim.conv2d(wrap_pad(cnv2_1, 1, 1), ngf * 4, [3, 3], scope='conv2_2', stride=2, padding="VALID")

      cnv3_1 = slim.conv2d(wrap_pad(cnv2_2, 1, 1), ngf * 4, [3, 3], scope='conv3_1', stride=1, padding="VALID")

      cnv3_2 = slim.conv2d(wrap_pad(cnv3_1, 1, 1), ngf * 4, [3, 3], scope='conv3_2', stride=1, padding="VALID")

      cnv3_3 = slim.conv2d(wrap_pad(cnv3_2, 1, 1), ngf * 8, [3, 3], scope='conv3_3', stride=2, padding="VALID")

      cnv4_1 = slim.conv2d(
         wrap_pad(cnv3_3, 2, 2), ngf * 8, [3, 3], scope='conv4_1', stride=1, rate=2, padding="VALID")
      cnv4_2 = slim.conv2d(
          wrap_pad(cnv4_1, 2, 2), ngf * 8, [3, 3], scope='conv4_2', stride=1, rate=2, padding="VALID")
      cnv4_3 = slim.conv2d(
          wrap_pad(cnv4_2, 2, 2), ngf * 8, [3, 3], scope='conv4_3', stride=1, rate=2, padding="VALID")

      # Adding skips
      skip = tf.concat([cnv4_3, cnv3_3], axis=3)
      cnv6_1 = slim.conv2d_transpose(
          wrap_pad(skip, 2, 2), ngf * 4, [4, 4], scope='conv6_1', stride=2, padding="VALID")
      # cnv6_1 = conv2d_transpose(skip, ngf * 4, [4,4], scope='conv6_1', stride=2, padding="VALID")
      cnv6_2 = slim.conv2d(wrap_pad(cnv6_1[:,5:-5,5:-5,:], 1, 1), ngf * 4, [3, 3], scope='conv6_2', stride=1, padding="VALID")
      cnv6_3 = slim.conv2d(wrap_pad(cnv6_2, 1, 1), ngf * 4, [3, 3], scope='conv6_3', stride=1, padding="VALID")

      skip = tf.concat([cnv6_3, cnv2_2], axis=3)
      cnv7_1 = slim.conv2d_transpose(
          wrap_pad(skip, 2, 2), ngf * 2, [4, 4], scope='conv7_1', stride=2, padding="VALID")
      # cnv7_1 = conv2d_transpose(skip, ngf * 2, [4,4], scope='conv7_1', stride=2, padding="VALID")
      cnv7_2 = slim.conv2d(wrap_pad(cnv7_1[:,5:-5,5:-5,:], 1, 1), ngf * 2, [3, 3], scope='conv7_2', stride=1, padding="VALID")

      skip = tf.concat([cnv7_2, cnv1_2], axis=3)
      cnv8_1 = slim.conv2d_transpose(
          wrap_pad(skip, 2, 2), ngf, [4, 4], scope='conv8_1', stride=2, padding="VALID")
      # cnv8_1 = conv2d_transpose(skip, ngf, [4,4], scope='conv8_1', stride=2, padding="VALID")
      cnv8_2 = slim.conv2d(wrap_pad(cnv8_1[:,5:-5,5:-5,:], 1, 1), ngf, [3, 3], scope='conv8_2', stride=1, padding="VALID")
      feat = cnv8_2
      pred = slim.conv2d(
          feat,
          num_outputs, [1, 1],
          stride=1,
          activation_fn=tf.nn.tanh,
          normalizer_fn=None,
          scope='color_pred')

  ## for exporting aaron's model
  msi_output = pred
  if FLAGS.net_only and FLAGS.which_color_pred == 'blend_psv':
    msi_output = tf.transpose(msi_output, [0, 3, 1, 2])
    msi_output = msi_output[:, :64, :, :]
    msi_output = tf.reshape(msi_output, [1, 8, 4, FLAGS.height, FLAGS.width])
    msi_output = tf.transpose(msi_output, [0, 1, 3, 2, 4])
    msi_output = tf.reshape(msi_output, [1, 8 * FLAGS.height, 4 * FLAGS.width], name='msi_output')
    return msi_output
  elif FLAGS.net_only and FLAGS.which_color_pred == 'alpha_only':
    msi_output = tf.transpose(msi_output, [0, 3, 1, 2])
    msi_output = msi_output[:, :, :, :]
    msi_output = tf.reshape(msi_output, [1, 8, 4, FLAGS.height, FLAGS.width])
    msi_output = tf.transpose(msi_output, [0, 1, 3, 2, 4])
    msi_output = tf.reshape(msi_output, [1, 8 * FLAGS.height, 4 * FLAGS.width], name='msi_output')
    return msi_output
  else:
    return pred

def msi_coord_train_net(inputs, num_outputs, ngf=64, vscope='net', reuse_weights=False):
  """Network definition for multiplane image (msi) inference.

  Args:
    inputs: stack of input images [batch, height, width, input_channels]
    num_outputs: number of output channels
    ngf: number of features for the first conv layer
    vscope: variable scope
    reuse_weights: whether to reuse weights (for weight sharing)
  Returns:
    pred: network output at the same spatial resolution as the inputs.
  """
  with tf.variable_scope(vscope, reuse=reuse_weights):
    with slim.arg_scope(
        [slim.conv2d, slim.conv2d_transpose], normalizer_fn=slim.layer_norm):
      cnv1_1 = coord_conv2d(inputs, ngf, [3, 3], scope='conv1_1', stride=1)
      cnv1_2 = coord_conv2d(cnv1_1, ngf * 2, [3, 3], scope='conv1_2', stride=2)
      cnv2_1 = coord_conv2d(cnv1_2, ngf * 2, [3, 3], scope='conv2_1', stride=1)
      cnv2_2 = coord_conv2d(cnv2_1, ngf * 4, [3, 3], scope='conv2_2', stride=2)
      cnv3_1 = coord_conv2d(cnv2_2, ngf * 4, [3, 3], scope='conv3_1', stride=1)
      cnv3_2 = coord_conv2d(cnv3_1, ngf * 4, [3, 3], scope='conv3_2', stride=1)
      cnv3_3 = coord_conv2d(cnv3_2, ngf * 8, [3, 3], scope='conv3_3', stride=2)
      cnv4_1 = coord_conv2d(cnv3_3, ngf * 8, [3, 3], scope='conv4_1', stride=1, rate=2)
      cnv4_2 = coord_conv2d(cnv4_1, ngf * 8, [3, 3], scope='conv4_2', stride=1, rate=2)
      cnv4_3 = coord_conv2d(cnv4_2, ngf * 8, [3, 3], scope='conv4_3', stride=1, rate=2)

      # Adding skips
      skip = tf.concat([cnv4_3, cnv3_3], axis=3)
      cnv6_1 = slim.conv2d_transpose(skip, ngf * 4, [4, 4], scope='conv6_1', stride=2)
      cnv6_2 = coord_conv2d(cnv6_1, ngf * 4, [3, 3], scope='conv6_2', stride=1)
      cnv6_3 = coord_conv2d(cnv6_2, ngf * 4, [3, 3], scope='conv6_3', stride=1)
      skip = tf.concat([cnv6_3, cnv2_2], axis=3)
      cnv7_1 =  slim.conv2d_transpose(skip, ngf * 2, [4, 4], scope='conv7_1', stride=2)
      cnv7_2 = coord_conv2d(cnv7_1, ngf * 2, [3, 3], scope='conv7_2', stride=1)
      skip = tf.concat([cnv7_2, cnv1_2], axis=3)
      cnv8_1 =  slim.conv2d_transpose(skip, ngf, [4, 4], scope='conv8_1', stride=2)
      cnv8_2 = coord_conv2d(cnv8_1, ngf, [3, 3], scope='conv8_2', stride=1)
      feat = cnv8_2
      pred = slim.conv2d(
          feat,
          num_outputs, [1, 1],
          stride=1,
          activation_fn=tf.nn.tanh,
          normalizer_fn=None,
          scope='color_pred')

  ## for exporting aaron's model
  msi_output = pred
  if FLAGS.net_only and FLAGS.which_color_pred == 'blend_psv':
    msi_output = tf.transpose(msi_output, [0, 3, 1, 2])
    msi_output = msi_output[:, :64, :, :]
    msi_output = tf.reshape(msi_output, [1, 8, 4, FLAGS.height, FLAGS.width])
    msi_output = tf.transpose(msi_output, [0, 1, 3, 2, 4])
    msi_output = tf.reshape(msi_output, [1, 8 * FLAGS.height, 4 * FLAGS.width], name='msi_output')
    return msi_output
  elif FLAGS.net_only and FLAGS.which_color_pred == 'alpha_only':
    msi_output = tf.transpose(msi_output, [0, 3, 1, 2])
    msi_output = msi_output[:, :, :, :]
    msi_output = tf.reshape(msi_output, [1, 8, 4, FLAGS.height, FLAGS.width])
    msi_output = tf.transpose(msi_output, [0, 1, 3, 2, 4])
    msi_output = tf.reshape(msi_output, [1, 8 * FLAGS.height, 4 * FLAGS.width], name='msi_output')
    return msi_output
  else:
    return pred

def msi_coord_inference_net(inputs, num_outputs, ngf=64, vscope='net', reuse_weights=False):
  """Network definition for multiplane image (msi) inference.

  Args:
    inputs: stack of input images [batch, height, width, input_channels]
    num_outputs: number of output channels
    ngf: number of features for the first conv layer
    vscope: variable scope
    reuse_weights: whether to reuse weights (for weight sharing)
  Returns:
    pred: network output at the same spatial resolution as the inputs.
  """

  if FLAGS.net_only:
     inputs = to_placeholder(inputs, name='plane_sweep_input')
     df = 'NHWC'
     pd = 'VALID'
  else:
     df = 'NHWC'
     pd = 'VALID'

  with tf.variable_scope(vscope, reuse=reuse_weights):
    with slim.arg_scope(
        [slim.conv2d, slim.conv2d_transpose], normalizer_fn=slim.layer_norm):
      # Encoder
      cnv1_1 = coord_inf_conv2d(inputs, ngf, [3, 3], scope='conv1_1', stride=1, padding=[1, 1, 1, 1])
      cnv1_2 = coord_inf_conv2d(cnv1_1, ngf * 2, [3, 3], scope='conv1_2', stride=2, padding=[1, 1, 1, 1])
      cnv2_1 = coord_inf_conv2d(cnv1_2, ngf * 2, [3, 3], scope='conv2_1', stride=1, padding=[1, 1, 1, 1])
      cnv2_2 = coord_inf_conv2d(cnv2_1, ngf * 4, [3, 3], scope='conv2_2', stride=2, padding=[1, 1, 1, 1])
      cnv3_1 = coord_inf_conv2d(cnv2_2, ngf * 4, [3, 3], scope='conv3_1', stride=1, padding=[1, 1, 1, 1])
      cnv3_2 = coord_inf_conv2d(cnv3_1, ngf * 4, [3, 3], scope='conv3_2', stride=1, padding=[1, 1, 1, 1])
      cnv3_3 = coord_inf_conv2d(cnv3_2, ngf * 8, [3, 3], scope='conv3_3', stride=2, padding=[1, 1, 1, 1])
      cnv4_1 = coord_inf_conv2d(cnv3_3, ngf * 8, [3, 3], scope='conv4_1', stride=1, rate=2, padding=[2, 3, 2, 3], slice=[0, int(FLAGS.height / 8), 0, int(FLAGS.width / 8)])
      cnv4_2 = coord_inf_conv2d(cnv4_1, ngf * 8, [3, 3], scope='conv4_2', stride=1, rate=2, padding=[2, 3, 2, 3], slice=[0, int(FLAGS.height / 8), 0, int(FLAGS.width / 8)])
      cnv4_3 = coord_inf_conv2d(cnv4_2, ngf * 8, [3, 3], scope='conv4_3', stride=1, rate=2, padding=[2, 3, 2, 3], slice=[0, int(FLAGS.height / 8), 0, int(FLAGS.width / 8)])
      # Decoder
      skip = tf.concat([cnv4_3, cnv3_3], axis=3)
      if FLAGS.smoothed:
        cnv6_1 = conv2d_transpose(skip, ngf * 4, [4, 4], scope='conv6_1', stride=2, padding=[1, 2, 1, 2])
      else:
        cnv6_1 = conv2d_transpose(skip, ngf * 4, [4, 4], scope='conv6_1', stride=2, slice=[2, int(FLAGS.height / 4) + 2, 2, int(FLAGS.width / 4) + 2])
      cnv6_2 = coord_inf_conv2d(cnv6_1, ngf * 4, [3, 3], scope='conv6_2', stride=1, padding=[1, 1, 1, 1])
      cnv6_3 = coord_inf_conv2d(cnv6_2, ngf * 4, [3, 3], scope='conv6_3', stride=1, padding=[1, 1, 1, 1])
      skip = tf.concat([cnv6_3, cnv2_2], axis=3)
      if FLAGS.smoothed:
        cnv7_1 = conv2d_transpose(skip, ngf * 2, [4, 4], scope='conv7_1', stride=2, padding=[1, 2, 1, 2])
      else:
        cnv7_1 = conv2d_transpose(skip, ngf * 2, [4, 4], scope='conv7_1', stride=2, slice=[2, int(FLAGS.height / 2) + 2, 2, int(FLAGS.width / 2) + 2])
      cnv7_2 = coord_inf_conv2d(cnv7_1, ngf * 2, [3, 3], scope='conv7_2', stride=1, padding=[1, 1, 1, 1])
      skip = tf.concat([cnv7_2, cnv1_2], axis=3)
      if FLAGS.smoothed:
        cnv8_1 = conv2d_transpose(skip, ngf, [4, 4], scope='conv8_1', stride=2, padding=[1, 2, 1, 2])
      else:
        cnv8_1 = conv2d_transpose(skip, ngf, [4, 4], scope='conv8_1', stride=2, slice=[2, int(FLAGS.height) + 2, 2, int(FLAGS.width) + 2])
      cnv8_2 = coord_inf_conv2d(cnv8_1, ngf, [3, 3], scope='conv8_2', stride=1, padding=[1, 1, 1, 1])
      feat = cnv8_2
      pred = conv2d(
          feat,
          num_outputs, [1, 1],
          stride=1,
          activation_fn=tf.nn.tanh,
          normalizer_fn=None,
          scope='color_pred')

  ## for exporting aaron's model
  msi_output = pred
  if FLAGS.net_only and FLAGS.which_color_pred == 'blend_psv':
      msi_output = tf.transpose(msi_output, [0, 3, 1, 2])
      output_h = 8
      msi_output = msi_output[:, :64, :, :]
      msi_output = tf.reshape(msi_output, [1, 8, output_h, FLAGS.height, FLAGS.width])
      msi_output = tf.transpose(msi_output, [0, 1, 3, 2, 4])
      msi_output = tf.reshape(msi_output, [1, 8 * FLAGS.height, output_h * FLAGS.width], name='msi_output')
  elif FLAGS.net_only and FLAGS.which_color_pred == 'alpha_only':
      msi_output = tf.transpose(msi_output, [0, 3, 1, 2])
      msi_output = msi_output[:, :, :, :]
      msi_output = tf.reshape(msi_output, [1, 8, 4, FLAGS.height, FLAGS.width])
      msi_output = tf.transpose(msi_output, [0, 1, 3, 2, 4])
      msi_output = tf.reshape(msi_output, [1, 8 * FLAGS.height, 4 * FLAGS.width], name='msi_output')
  else:
      msi_output = tf.identity(msi_output, name='msi_output')

  return msi_output


############################codes from Pixel2Mesh paper #########################################
#################################################################################################
#  Copyright (C) 2019 Nanyang Wang, Yinda Zhang, Zhuwen Li, Yanwei Fu, Wei Liu, Yu-Gang Jiang, Fudan University
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def conv_graph(inputs, input_dim, output_dim, support, scope=None, activation_fn=tf.nn.relu,
                bias=True, gcn_block_id=1, dropout=False, sparse_inputs=False, featureless=False):

    with tf.name_scope(scope) as my_scope:
        conv = conv_graph_helper(inputs, input_dim, output_dim, support, scope=scope)

        if activation_fn != None:
            conv = activation_fn(conv)

    return conv

def conv_graph_helper(inputs, input_dim, output_dim, support, bias=True, scope=None, sparse_inputs=False, featureless=False):
    '''@From P2M'''
    with tf.variable_scope(scope + '_vars'):
        kernel = {}
        for i in range(len(support)):
            kernel[i] = glorot([input_dim, output_dim], name='weights_' + str(i))

        if bias:
            bias = zeros([output_dim], name='bias')

    x = inputs
    # convolve
    supports = list()

    for i in range(len(support)):
        if not featureless:
            pre_sup = dot(x, kernel[i],
                          sparse=sparse_inputs)
        else:
            pre_sup = kernel[i]

        curr_support = dot(support[i], pre_sup, sparse=True)
        supports.append(curr_support)
    output = tf.add_n(supports)

    # bias
    if bias:
        output += bias

    return output

def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def gcn_net(inputs, num_outputs, support, ngf=64, vscope='net', reuse_weights=False):
    print("using gcn!!!!")
    n_vertices, n_features = inputs.get_shape().as_list()

    with tf.variable_scope(vscope, reuse=reuse_weights):
        conv = conv_graph(inputs, input_dim = n_features, output_dim = ngf, support=support, scope='conv1_1',gcn_block_id=1 )
        for i in range(12):
            conv = conv_graph(conv, input_dim = ngf, output_dim = ngf, support = support, scope='conv2_'+str(i), gcn_block_id=1 )
        pred = conv_graph(conv, input_dim = ngf, output_dim = num_outputs, support = support,
                          scope='conv3_1', gcn_block_id=1, activation_fn=tf.nn.tanh)

    return pred