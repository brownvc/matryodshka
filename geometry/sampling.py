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


"""Module for bilinear sampling.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

def bilinear_wrapper(imgs, coords):
  """Wrapper around bilinear sampling function, handles arbitrary input sizes.
  Args:
    imgs: [..., H_s, W_s, C] images to resample
    coords: [..., H_t, W_t, 2], source pixel locations from which to copy
  Returns:
    [..., H_t, W_t, C] images after bilinear sampling from input.
  """
  # The bilinear sampling code only handles 4D input, so we'll need to reshape.
  init_dims = imgs.get_shape().as_list()[:-3:]
  end_dims_img = imgs.get_shape().as_list()[-3::]
  end_dims_coords = coords.get_shape().as_list()[-3::]
  prod_init_dims = init_dims[0]
  for ix in range(1, len(init_dims)):
    prod_init_dims *= init_dims[ix]

  imgs = tf.reshape(imgs, [prod_init_dims] + end_dims_img)
  coords = tf.reshape(
      coords, [prod_init_dims] + end_dims_coords)
  imgs_sampled = tf.contrib.resampler.resampler(imgs, coords)
  imgs_sampled = tf.reshape(
      imgs_sampled, init_dims + imgs_sampled.get_shape().as_list()[-3::])
  return imgs_sampled

def sph_bilinear_wrapper(imgs, coords):
  return sphere_resample(imgs, coords)

def bilinear_wrapper2(imgs, coords):
    """Wrapper around bilinear sampling function, handles arbitrary input sizes.
    Args:
      imgs: [..., H_s, W_s, C] images to resample
      coords: [..., H_t, W_t, 2], source pixel locations from which to copy
    Returns:
      [..., H_t, W_t, C] images after bilinear sampling from input.
    """
    return resample(imgs, coords)

def sphere_resample(image, pixels):
  print(pixels.get_shape().as_list())

  batch_size, _, n_vertex, _ = pixels.get_shape().as_list()
  _, height, width, channels = image.get_shape().as_list()
  image_shape = [batch_size, 1, n_vertex, channels]

  # Unstack and reshape
  pixels = tf.transpose(pixels, [0, 3, 1, 2])
  pixels = tf.reshape(pixels, [batch_size, 2, -1])

  x, y = tf.unstack(pixels, axis=1)
  x = tf.reshape(x, [-1])
  y = tf.reshape(y, [-1])

  # Four corners
  x = x
  y = y
  x0 = tf.cast(tf.floor(x), 'int32')
  x1 = x0 + 1
  y0 = tf.cast(tf.floor(y), 'int32')
  y1 = y0 + 1

  diff_x0 = x - tf.cast(x0, tf.float32)
  diff_y0 = y - tf.cast(y0, tf.float32)
  diff_x1 = tf.cast(x1, tf.float32) - x
  diff_y1 = tf.cast(y1, tf.float32) - y

  x0 = tf.mod(x0 + width, width)
  y0 = tf.mod(y0 + height, height)
  x1 = tf.mod(x1 + width, width)
  y1 = tf.mod(y1 + height, height)

  # Indices
  b = repeat_int(tf.range(batch_size), n_vertex)

  indices_a = tf.stack([b, y0, x0], axis=1)
  indices_b = tf.stack([b, y0, x1], axis=1)
  indices_c = tf.stack([b, y1, x0], axis=1)
  indices_d = tf.stack([b, y1, x1], axis=1)

  # Pixel values
  pixel_values_a = tf.gather_nd(image, indices_a)
  pixel_values_b = tf.gather_nd(image, indices_b)
  pixel_values_c = tf.gather_nd(image, indices_c)
  pixel_values_d = tf.gather_nd(image, indices_d)

  # Weights
  x0 = tf.cast(x0, tf.float32)
  x1 = tf.cast(x1, tf.float32)
  y0 = tf.cast(y0, tf.float32)
  y1 = tf.cast(y1, tf.float32)

  area_a = tf.expand_dims((diff_y1 * diff_x1), 1)
  area_b = tf.expand_dims((diff_y1 * diff_x0), 1)
  area_c = tf.expand_dims((diff_y0 * diff_x1), 1)
  area_d = tf.expand_dims((diff_y0 * diff_x0), 1)

  res = tf.add_n([area_a * pixel_values_a,
                  area_b * pixel_values_b,
                  area_c * pixel_values_c,
                  area_d * pixel_values_d])

  return tf.reshape(res, image_shape)


def resample(image, pixels):
  batch_size, pixels_height, pixels_width, _ = pixels.get_shape().as_list()
  _, height, width, channels = image.get_shape().as_list()
  image_shape = [batch_size, pixels_height, pixels_width, channels]

  # Unstack and reshape
  pixels = tf.transpose(pixels, [0, 3, 1, 2])
  pixels = tf.reshape(pixels, [batch_size, 2, -1])

  x, y = tf.unstack(pixels, axis=1)

  x = tf.reshape(x, [-1])
  y = tf.reshape(y, [-1])

  # Four corners
  x = x
  y = y
  x0 = tf.cast(tf.floor(x), 'int32')
  x1 = x0 + 1
  y0 = tf.cast(tf.floor(y), 'int32')
  y1 = y0 + 1

  diff_x0 = x - tf.cast(x0, tf.float32)
  diff_y0 = y - tf.cast(y0, tf.float32)
  diff_x1 = tf.cast(x1, tf.float32) - x
  diff_y1 = tf.cast(y1, tf.float32) - y

  x0 = tf.mod(x0 + width, width)
  y0 = tf.mod(y0 + height, height)
  x1 = tf.mod(x1 + width, width)
  y1 = tf.mod(y1 + height, height)

  # Indices
  b = repeat_int(tf.range(batch_size), pixels_height * pixels_width)

  indices_a = tf.stack([b, y0, x0], axis=1)
  indices_b = tf.stack([b, y0, x1], axis=1)
  indices_c = tf.stack([b, y1, x0], axis=1)
  indices_d = tf.stack([b, y1, x1], axis=1)

  # Pixel values
  pixel_values_a = tf.gather_nd(image, indices_a)
  pixel_values_b = tf.gather_nd(image, indices_b)
  pixel_values_c = tf.gather_nd(image, indices_c)
  pixel_values_d = tf.gather_nd(image, indices_d)

  # Weights
  x0 = tf.cast(x0, tf.float32)
  x1 = tf.cast(x1, tf.float32)
  y0 = tf.cast(y0, tf.float32)
  y1 = tf.cast(y1, tf.float32)

  area_a = tf.expand_dims((diff_y1 * diff_x1), 1)
  area_b = tf.expand_dims((diff_y1 * diff_x0), 1)
  area_c = tf.expand_dims((diff_y0 * diff_x1), 1)
  area_d = tf.expand_dims((diff_y0 * diff_x0), 1)

  res = tf.add_n([area_a * pixel_values_a,
                  area_b * pixel_values_b,
                  area_c * pixel_values_c,
                  area_d * pixel_values_d])

  return tf.reshape(res, image_shape)

def repeat_int(x, num_repeats):
  x = tf.tile(tf.expand_dims(x, axis=1), [1, num_repeats])
  return tf.reshape(x, [-1])

