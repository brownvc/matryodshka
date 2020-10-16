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

"""
This script exports model into .pb file, which later get converted to onnx file for TensorRT deployment.
"""

from __future__ import division
import sys
sys.path.append('/usr/lib/python2.7/dist-packages')
import tensorflow as tf
from matryodshka.msi import MSI
from matryodshka.utils import build_matrix
from geometry.sampling import bilinear_wrapper
import numpy as np
import os
flags = tf.app.flags

# Input flags
flags.DEFINE_integer('width', 640, 'Image width')
flags.DEFINE_integer('height', 320, 'Image height')
flags.DEFINE_float('xoffset', 0.0,
                   'Camera x-offset from first to second image.')
flags.DEFINE_float('yoffset', 0.0,
                   'Camera y-offset from first to second image.')
flags.DEFINE_float('zoffset', 0.0,
                   'Camera z-offset from first to second image.')
flags.DEFINE_float('min_depth', 1, 'Minimum scene depth.')
flags.DEFINE_float('max_depth', 100, 'Maximum scene depth.')
flags.DEFINE_integer(
    'xshift', 0, 'Horizontal pixel shift for image2 '
    '(i.e., difference in x-coordinate of principal point '
    'from image2 to image1).')
flags.DEFINE_integer(
    'yshift', 0, 'Vertical pixel shift for image2 '
    '(i.e., difference in y-coordinate of principal point '
    'from image2 to image1).')
flags.DEFINE_string('pose1', '',
                    ('Camera pose for first image (if not identity).'
                     ' Twelve space- or comma-separated floats, forming a 3x4'
                     ' matrix in row-major order.'))
flags.DEFINE_string('pose2', '',
                    ('Pose for second image (if not identity).'
                     ' Twelve space- or comma-separated floats, forming a 3x4'
                     ' matrix in row-major order. If pose2 is specified, then'
                     ' xoffset/yoffset/zoffset flags will be used for rendering'
                     ' output views only.'))
flags.DEFINE_string('remap_ref', '',
                    ('Remap file for reference image.'))
flags.DEFINE_string('remap_src', '',
                    ('Remap file for source image.'))

# Output flags
flags.DEFINE_string('test_outputs', '',
                    'Which outputs to save. Can concat the following with "_": '
                    '[src_image, ref_image, tgt_image, psp (for perspective crop), hres_tgt_image, '
                    ' src_output_image, ref_output_image, psv, alphas, blend_weights, rgba_layers].')

# Model flags. Defaults are the model described in the SIGGRAPH 2018 paper.  See
# README for more details.
flags.DEFINE_string('model_root', 'checkpoints/',
                    'Root directory for model checkpoints.')
flags.DEFINE_string('model_name', 'ods-wotemp-elpips-coord',
                    'Name of the model to use for inference.')
flags.DEFINE_string('which_color_pred', 'blend_psv',
                    'Color output format: [blend_psv, blend_bg, blend_bg_psv, alpha_only].')
flags.DEFINE_integer('num_psv_planes', 32, 'Number of planes for PSV.')
flags.DEFINE_integer('num_msi_planes', 32, 'Number of msi planes to infer.')
flags.DEFINE_integer('ngf', 64, 'Number of filters.')
flags.DEFINE_string('pb_output','matryodshka','name of the pb file')

# Graph export settings
flags.DEFINE_boolean('clip', False,
                     'Clip weights by float16 range.')
flags.DEFINE_boolean('flip_y', False,
                     'Flip y axis in input images')
flags.DEFINE_boolean('flip_channels', False,
                     'Flip channels in input image')
flags.DEFINE_boolean('rgba', False,
                     'Is image rgba')
flags.DEFINE_boolean('remap', False,
                     'Whether or not to remap')
flags.DEFINE_boolean('net_only', False,
                     'Extract only the network')
flags.DEFINE_boolean('smoothed', False,
                     'Smooth conv2d transpose ops')
flags.DEFINE_boolean('jitter', False, 'jitter for transform inverse traning.')

# Camera models, input, output, internal MSI representation
flags.DEFINE_string('input_type', 'ODS',
                    'Input image type. [PP, ODS]')
flags.DEFINE_string('operation', 'train',
                    'Which operation to perform. [train, export]')
flags.DEFINE_string('supervision', 'tgt', "Images to supervise on. [tgt, ref, src, hrestgt] concatenated with _")
flags.DEFINE_boolean('transform_inverse_reg', False, 'Whether to train with transform-inverse regularization.')
flags.DEFINE_boolean('coord_net', False, 'Whether to append CoordNet during convolution.')

# Set flags
FLAGS = flags.FLAGS

def crop_to_multiple(image, size):
  """Crop image to a multiple of size in height and width."""
  # Compute how much we need to remove.
  shape = image.get_shape().as_list()
  height = shape[0]
  width = shape[1]
  new_width = width - (width % size)
  new_height = height - (height % size)

  # Crop amounts. Extra pixel goes on the left side.
  left = (width % size) // 2
  right = new_width + left
  top = (height % size) // 2
  bottom = new_height + top

  return image[top:bottom, left:right, :]

def process_image(raw, height, width, channels, padx, pady, remap_file):
  """Load an image, pad, and shift it."""
  image = tf.reshape(raw, (height, width, channels))

  # Extract rgb
  if FLAGS.rgba:
    image = image[:, :, :3]

  # Convert image to float32, 0-1 range
  image = tf.image.convert_image_dtype(image, tf.float32)

  # Remap image
  if FLAGS.remap:
    image = remap_image(image, remap_file)

  # Flip y
  if FLAGS.flip_y:
      image = tf.reverse(image, axis=[0])

  # Flip channels
  if FLAGS.flip_channels:
      image = tf.reverse(image, axis=[2])

  # Pad
  image = tf.pad(image, [[pady, pady], [padx, padx], [0, 0]])
  image.set_shape([None, None, 3])  # RGB images have 3 channels.

  return image

def remap_image(image, remap_file):
    remap_vals = np.load(remap_file)
    remap_tensor = tf.expand_dims(tf.convert_to_tensor(remap_vals), 0)
    image = tf.expand_dims(image, 0)
    return tf.squeeze(bilinear_wrapper(image, remap_tensor))

def pose_from_flag(flag):
  if flag:
    values = [float(x) for x in flag.replace(',', ' ').split()]
    assert len(values) == 12
    return [values[0:4], values[4:8], values[8:12], [0.0, 0.0, 0.0, 1.0]]
  else:
    return [[1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]

def get_inputs(padx, pady, width, height):
  """Get images, poses and intrinsics in required format."""

  inputs = {}

  # Process images
  channels = 4 if FLAGS.rgba else 3
  image1 = tf.placeholder(tf.uint8, (width * height * channels), name='ref_image')
  image2 = tf.placeholder(tf.uint8, (width * height * channels), name='src_image')

  with tf.name_scope('process_image'):
      image1 = process_image(image1, height, width, channels, padx, pady, FLAGS.remap_ref)
      image2 = process_image(image2, height, width, channels, padx, pady, FLAGS.remap_src)

  # Images pad and crop
  shape1_before_crop = tf.shape(image1)
  shape2_before_crop = tf.shape(image2)
  image1 = crop_to_multiple(image1, 16)
  image2 = crop_to_multiple(image2, 16)
  shape1_after_crop = tf.shape(image1)
  shape2_after_crop = tf.shape(image2)

  with tf.control_dependencies([
      tf.Assert(
          tf.reduce_all(
              tf.logical_and(
                  tf.equal(shape1_before_crop, shape2_before_crop),
                  tf.equal(shape1_after_crop, shape2_after_crop))), [
                      'Shape mismatch:', shape1_before_crop, shape2_before_crop,
                      shape1_after_crop, shape2_after_crop
                  ])
  ]):

    # Add batch dimension (size 1).
    image1 = tf.expand_dims(image1, 0)
    image2 = tf.expand_dims(image2, 0)

  # Poses
  pose_one = pose_from_flag(FLAGS.pose1)
  pose_two = pose_from_flag(FLAGS.pose2)
  with tf.name_scope('build_matrices'):
    pose_one = build_matrix(pose_one)
    pose_two = build_matrix(pose_two)
    pose_one = tf.expand_dims(pose_one, 0)
    pose_two = tf.expand_dims(pose_two, 0)
    intrinsics = build_matrix([[0.032, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0]])
    intrinsics = tf.expand_dims(intrinsics, 0)

  # Set inputs
  inputs['ref_image'] = image1
  inputs['src_image'] = image2
  inputs['ref_pose'] = pose_one
  inputs['src_pose'] = pose_two
  inputs['intrinsics'] = intrinsics

  # Second order inputs
  inputs['ref_pose_inv'] = tf.matrix_inverse(inputs['ref_pose'], name='ref_pose_inv')
  inputs['src_pose_inv'] = tf.matrix_inverse(inputs['src_pose'], name='src_pose_inv')
  inputs['intrinsics_inv'] = tf.matrix_inverse(inputs['intrinsics'], name='intrinsics_inv')

  raw_hres_tgt_image = None
  raw_hres_ref_image = None
  raw_hres_src_images = None
  inputs['hres_ref'] = raw_hres_ref_image
  inputs['hres_src'] = raw_hres_src_images
  inputs['hres_tgt'] = raw_hres_tgt_image

  return inputs

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
  """
  Freezes the state of a session into a pruned computation graph.
  Creates a new computation graph where variable nodes are replaced by
  constants taking their current value in the session. The new graph will be
  pruned so subgraphs that are not necessary to compute the requested
  outputs are removed.
  @param session The TensorFlow session to be frozen.
  @param keep_var_names A list of variable names that should not be frozen,
                        or None to freeze all the variables in the graph.
  @param output_names Names of the relevant graph outputs.
  @param clear_devices Remove the device directives from the graph for better portability.
  @return The frozen graph definition.
  """
  graph = session.graph

  with graph.as_default():
    freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
    output_names = output_names or []
    output_names += [v.op.name for v in tf.global_variables()]
    input_graph_def = graph.as_graph_def()

    if clear_devices:
      for node in input_graph_def.node:
          node.device = ""

    frozen_graph = tf.graph_util.convert_variables_to_constants(
      session, input_graph_def, output_names, freeze_var_names)

    return frozen_graph

def main(_):

  # Get inputs
  pady = 0
  padx = 0
  inputs = get_inputs(padx, pady, FLAGS.width, FLAGS.height)

  # Build the network
  model = MSI()
  psv_planes = model.inv_depths(FLAGS.min_depth, FLAGS.max_depth,
                              FLAGS.num_psv_planes)
  msi_planes = model.inv_depths(FLAGS.min_depth, FLAGS.max_depth,
                              FLAGS.num_msi_planes)

  outputs = model.infer_msi(
      inputs['src_image'], inputs['ref_image'], inputs['hres_src'], inputs['hres_ref'],
      inputs['ref_pose'], inputs['src_pose'], inputs['intrinsics'],
      FLAGS.which_color_pred, FLAGS.num_msi_planes, psv_planes, FLAGS.test_outputs, ngf=FLAGS.ngf)

  # Load weights and save graph
  saver = tf.train.Saver([var for var in tf.trainable_variables()])
  ckpt_dir = os.path.join(FLAGS.model_root, FLAGS.model_name)
  ckpt_file = tf.train.latest_checkpoint(ckpt_dir)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    with sess.as_default():
      saver.restore(sess, ckpt_file)
      if FLAGS.clip:
          for var in tf.trainable_variables():
              tv = sess.graph.get_tensor_by_name(var.name)
              nv = sess.run(tv)
              if np.amax(nv) > tf.float16.max or np.amin(nv) < tf.float16.min:
                  print(var.name)
                  print(np.amin(nv))

              clipped_nv = np.clip(nv, tf.float16.min, tf.float16.max)
              #clipped_v = np.zeros_like(sess.run(v))
              sess.run(tf.assign(tv, clipped_nv))

      frozen = freeze_session(sess, output_names=['msi_output'], clear_devices=False)

      # Write out
      with tf.gfile.GFile('export/%s.pb' % FLAGS.pb_output, "wb") as f:
          f.write(frozen.SerializeToString())


if __name__ == '__main__':
  tf.app.run()