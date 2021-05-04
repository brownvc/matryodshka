#!/usr/bin/python
#
# Copyright 2020 Brown Visual Computing Lab / Authors of the accompanying paper Matryodshka 
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

"""Main script for training."""

from __future__ import division
import tensorflow as tf
from matryodshka.data_loader import ReplicaSequenceDataLoader, RealEstateSequenceDataLoader
from matryodshka.msi import MSI
from matryodshka.utils import interpolate_pose, load_mesh_input
from geometry.spherical import tf_random_rotation
import numpy as np

# Note that the flags below are a subset of all flags. The remainder (data
# loading relevant) are defined in loader.py.
flags = tf.app.flags

#i/o
flags.DEFINE_string('cameras_glob', 'glob/train/ods/*.txt',
                    'Glob string for training set camera files.')
flags.DEFINE_string('image_dir', '/path/to/train_640x320',
                    'Path to training image directories.')
flags.DEFINE_string('hres_image_dir', '/path/to/train_4096x2048',
                    'Path to high-resolution training image directories.')
flags.DEFINE_string('checkpoint_dir', 'checkpoints',
                    'Location to save the models.')
flags.DEFINE_string('experiment_name', '', 'Name for the experiment to run.')
flags.DEFINE_integer('shuffle_seq_length', 3, 'Number of images for each input group.')
flags.DEFINE_integer('num_synth', 2, 'Number of additional frames for synthesis.')

# training hyper-parameters
flags.DEFINE_float('learning_rate', 0.0002, 'Learning rate')
flags.DEFINE_float('beta1', 0.9, 'beta1 hyperparameter for Adam optimizer.')
flags.DEFINE_integer('random_seed', 8964, 'Random seed.')
flags.DEFINE_integer('max_steps', 10000000, 'Maximum number of training steps.')
flags.DEFINE_integer('summary_freq', 50, 'Logging frequency.')
flags.DEFINE_integer('save_latest_freq', 2000, 'Frequency with which to save the model (overwrites previous model).')
flags.DEFINE_boolean('continue_train', False, 'Continue training from previous checkpoint.')

# model-related
flags.DEFINE_string('operation', 'train', 'Which operation to perform. [train, export]')
flags.DEFINE_string('input_type', 'ODS', 'Input image type. [PP, ODS]')
flags.DEFINE_boolean('coord_net', False, 'Whether to append CoordNet during convolution.')
flags.DEFINE_boolean('transform_inverse_reg', False, 'Whether to train with transform-inverse regularization.')
flags.DEFINE_boolean('jitter', False, 'Jitter rotation')
flags.DEFINE_string('which_color_pred', 'blend_psv',
                    'Color output format: [blend_psv, blend_bg, blend_bg_psv, alpha_only].')
flags.DEFINE_integer('ngf', 64, 'Number of filters.')
flags.DEFINE_float('min_depth', 1, 'Minimum scene depth.')
flags.DEFINE_float('max_depth', 100, 'Maximum scene depth.')
flags.DEFINE_integer('num_psv_planes', 32, 'Number of planes for plane sweep volume (PSV).')
flags.DEFINE_integer('num_msi_planes', 32, 'Number of msi planes to predict.')

# loss-related
flags.DEFINE_string('which_loss', 'pixel', 'Which loss to use to compare rendered and ground truth images. '
                                           'Can be "pixel" or "elpips".')
flags.DEFINE_boolean('spherical_attention', False, 'Calculate loss with a spherically aware weight map.')

# for model export
flags.DEFINE_boolean('net_only', False,
                     'Extract only the network')
flags.DEFINE_boolean('smoothed', False,
                     'Smooth conv2d transpose ops')

# for saving relevant inputs/outputs without starting training
flags.DEFINE_boolean('dry_run',False,'Dry run to save images without inference. [src, ref, tgt images + formatted psv]')
flags.DEFINE_boolean('dry_run_inference',False,'Dry run to save images with inference '
                                               '[src, ref, tgt images + formatted psv + predicted rgba layers].')

# experiments Flags
flags.DEFINE_boolean('wreg', False, 'Add weight regularization.')
flags.DEFINE_boolean('mixed_precision', False, "Enable mixed precision training.")
flags.DEFINE_string('supervision', 'tgt', "Images to supervise on. [tgt, ref, src, hrestgt] concatenated with _")
flags.DEFINE_float('rot_factor',1.0,'for experiments with rotation jittering range, n x 0.03 radians')
flags.DEFINE_float('tr_factor',1.0,'for experiments with translation jittering range, n x 0.01')

flags.DEFINE_boolean('gcn',False,'Train with gcn.')
flags.DEFINE_integer('subdiv',7,'subdivision level for the spherical mesh we want to operate on.')

FLAGS = flags.FLAGS
def main(_):
    
  if FLAGS.input_type == 'PP': assert('hrestgt' not in FLAGS.supervision)

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.set_random_seed(FLAGS.random_seed)
  FLAGS.checkpoint_dir += '/%s/' % FLAGS.experiment_name
  if not tf.gfile.IsDirectory(FLAGS.checkpoint_dir):
    tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

  tf.logging.info("Image dir: %s" % FLAGS.image_dir)
  if 'hrestgt' in FLAGS.supervision: tf.logging.info("High-resolution image dir: %s" % FLAGS.image_dir)

  if FLAGS.input_type == 'REALESTATE_PP':
      data_loader = RealEstateSequenceDataLoader(FLAGS.cameras_glob, FLAGS.image_dir, True,
                                                 1, FLAGS.shuffle_seq_length, FLAGS.random_seed)
  else:
      data_loader = ReplicaSequenceDataLoader(FLAGS.cameras_glob, FLAGS.image_dir, FLAGS.hres_image_dir, True,
                                              FLAGS.shuffle_seq_length, FLAGS.random_seed)
  train_batch = data_loader.sample_batch()

  # additional input tensors
  if 'PP' in FLAGS.input_type:
      train_batch['interp_pose']  = interpolate_pose(train_batch["ref_pose"], train_batch["src_pose"])
      train_batch['interp_pose_inv'] = tf.matrix_inverse(train_batch["interp_pose"],name='interp_pose_inv')
  train_batch['intrinsics_inv'] = tf.matrix_inverse(train_batch['intrinsics'], name='intrinsics_inv')
  train_batch['ref_pose_inv'] = tf.matrix_inverse(train_batch['ref_pose'], name='ref_pose_inv')
  train_batch['jitter_pose'] = tf.identity(tf_random_rotation(FLAGS.rot_factor, FLAGS.tr_factor), name='jitter_pose')
  train_batch['jitter_pose_inv'] = tf.matrix_inverse(train_batch['jitter_pose'], name='jitter_pose_inv')

  if FLAGS.gcn:
      coord, support, pix2vertex_lookup = load_mesh_input()
      train_batch['p2v'] = tf.dtypes.cast(tf.identity(pix2vertex_lookup, name='p2v'), dtype=tf.float32)
      train_batch['coord'] = tf.identity(coord, name='coord')
      train_batch['support'] = [tf.SparseTensor(indices=np.asarray(support[i][0], dtype=np.float32),
                                                values=np.asarray(support[i][1], dtype=np.float32),
                                                dense_shape=np.asarray(support[i][2], dtype=np.float32)) for i in range(len(support))]

  model = MSI()
  train_op, rgba_layers, orig_pose, raw_tgt_image, first_tgt_pose = model.build_train_graph(train_batch, FLAGS.min_depth, FLAGS.max_depth, FLAGS.num_psv_planes,
                                     FLAGS.num_msi_planes, FLAGS.which_color_pred, FLAGS.which_loss,
                                     FLAGS.learning_rate, FLAGS.beta1)
  model.train(train_op, rgba_layers, orig_pose, raw_tgt_image, first_tgt_pose, FLAGS.checkpoint_dir, FLAGS.continue_train,
              FLAGS.summary_freq, FLAGS.save_latest_freq, FLAGS.max_steps)

if __name__ == '__main__':
  tf.app.run()
