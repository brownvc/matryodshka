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
Class definition of the data loader.
"""
from __future__ import division
import tensorflow as tf
from tensorflow import flags
import loader

FLAGS = flags.FLAGS

class ReplicaSequenceDataLoader(object):
  """Loader for 360 data."""
  def __init__(self,
               cameras_glob='glob/train/ods/*.txt',
               image_dir='train/',
               hres_image_dir='train/',
               training=True,
               seq_length=3,
               random_seed=8964,
               map_function=None,
               repeat_sample=None):

    self.random_seed = random_seed
    self.seq_length = seq_length
    self.batch_size = FLAGS.batch_size
    self.image_height = FLAGS.height
    self.image_width = FLAGS.width
    self.hres_image_height = FLAGS.hres_height
    self.hres_image_width = FLAGS.hres_width
    self.load_hres = 'hrestgt' in FLAGS.supervision
    self.repeat_sample = repeat_sample
    self.datasets = loader.create_loader_from_flags(
        cameras_glob=cameras_glob,
        image_dir=image_dir,
        hres_image_dir = hres_image_dir,
        training=training,
        map_function=map_function)

  def set_shapes(self, examples):
    """Set static shapes of the mini-batch of examples.
    Args:
      examples: a batch of examples
    Returns:
      examples with correct static shapes
    """
    b = self.batch_size
    h = self.image_height
    w = self.image_width

    if FLAGS.input_type == 'ODS':
        hh = self.hres_image_height
        hw = self.hres_image_width

        examples['tgt_image'].set_shape([b, h, w, 3])
        examples['ref_image'].set_shape([b, h, w, 3])
        examples['src_image'].set_shape([b, h, w, 3])
        examples['src_pose'].set_shape([b, 4, 4])
        examples['ref_pose'].set_shape([b, 4, 4])
        examples['tgt_pose'].set_shape([b, 3])
        examples['intrinsics'].set_shape([b, 3, 3])
        if self.load_hres:
            examples['hres_tgt_image'].set_shape([b, hh, hw, 3])
            examples['hres_ref_image'].set_shape([b, hh, hw, 3])
            examples['hres_src_image'].set_shape([b, hh, hw, 3])

    elif FLAGS.input_type == 'PP':

        examples['tgt_image'].set_shape([b, h, w, 3])
        examples['ref_image'].set_shape([b, h, w, 3])
        examples['src_image'].set_shape([b, h, w, 3])
        examples['src_pose'].set_shape([b, 4, 4])
        examples['tgt_pose'].set_shape([b, 4, 4])
        examples['ref_pose'].set_shape([b, 4, 4])
        examples['intrinsics'].set_shape([b, 3, 3])

    return examples

  def sample_batch(self):
    """
    Samples a batch of examples for training / testing.
    Returns:
      A batch of examples.
    """
    example = self.datasets.sequences.map(self.format_for_mpi())
    if self.repeat_sample != None:
        example = example.flat_map(lambda x: tf.data.Dataset.from_tensors(x).repeat(self.repeat_sample))
    iterator = example.make_one_shot_iterator()
    return self.set_shapes(iterator.get_next())

  def format_for_mpi(self):
    """
    Format the sampled sequence for MPI training/inference.
    """
    def make_intrinsics_matrix(fx, fy, cx, cy):
        # Assumes batch input.
        batch_size = fx.get_shape().as_list()[0]
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.constant([0., 0., 1.], shape=[1, 3])
        r3 = tf.tile(r3, [batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)
        return intrinsics

    def format_sequence(sequence):
      if FLAGS.input_type == 'ODS':
          images = sequence.image
          hres_images = sequence.hres_image
          images.set_shape([self.batch_size, self.seq_length, self.image_height, self.image_width, 3])
          if self.load_hres:
              hres_images.set_shape([self.batch_size, self.seq_length, self.hres_image_height, self.hres_image_width, 3])

          tgt_pos = sequence.tgt_pos
          tgt_pos.set_shape([self.batch_size, 3])
          ref_image = images[:, 0]
          src_image = images[:, 1]
          tgt_image = images[:, 2]
          if self.load_hres:
              hres_ref_image = hres_images[:, 0]
              hres_src_image = hres_images[:, 1]
              hres_tgt_image = hres_images[:, 2]

          scene_id = sequence.scene_id
          image_id = sequence.image_id

          # Pose is inverse of view matrix stored in ODS sequence
          pose_temp = sequence.pose_inv[0]
          pose = tf.linalg.inv(pose_temp)
          pose_one = tf.expand_dims(pose, 0)
          pose_one = tf.tile(pose_one,[self.batch_size,1,1])
          pose_two = pose_one

          # camera intrinsic matrix for perspective -> for equirect stores the baseline of stereo ods images
          intrinsics = [[sequence.baseline[0], 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
          intrinsics = tf.expand_dims(intrinsics,0)
          intrinsics = tf.tile(intrinsics,[self.batch_size,1,1])

          # Put everything into a dictionary.
          instance = {}
          instance['tgt_image'] = tgt_image
          instance['ref_image'] = ref_image
          instance['src_image'] = src_image
          if self.load_hres:
              instance['hres_tgt_image'] = hres_tgt_image
              instance['hres_ref_image'] = hres_ref_image
              instance['hres_src_image'] = hres_src_image
          instance['src_pose'] = pose_one
          instance['ref_pose'] = pose_two
          instance['tgt_pose'] = tgt_pos

          tgt_pose_rt = tf.tile(tf.expand_dims(tf.eye(3), axis=0), [self.batch_size, 1, 1])
          tgt_pose_rt = tf.concat([tgt_pose_rt, tf.expand_dims(tgt_pos, axis=-1)], axis=2)
          tgt_pose_rt = tf.concat([tgt_pose_rt, tf.expand_dims(tf.eye(4),0)[:,3:,:]], axis=1)
          instance['tgt_pose_rt'] = tgt_pose_rt

          instance['intrinsics'] = intrinsics
          instance['scene_id'] = scene_id
          instance['image_id'] = image_id
          return instance

      elif FLAGS.input_type == 'PP':

          images = sequence.image
          images.set_shape([
              self.batch_size, self.seq_length, self.image_height,
              self.image_width, 3
          ])

          ref_image = images[:, 0]
          src_image = images[:, 1]
          tgt_image = images[:, 2]

          scene_id = sequence.scene_id
          image_id = sequence.image_id

          pose_one = [[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]]
          pose_two = [[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]]
          pose_three = [[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]]

          pose_two[0][3] = -sequence.input_offset[0]
          pose_three[0][3] = -sequence.tgt_offset[0]
          pose_one = tf.expand_dims(pose_one, 0)
          pose_one = tf.tile(pose_one, [self.batch_size, 1, 1])
          pose_two = tf.expand_dims(pose_two, 0)
          pose_two = tf.tile(pose_two, [self.batch_size, 1, 1])
          pose_three = tf.expand_dims(pose_three, 0)
          pose_three = tf.tile(pose_three, [self.batch_size, 1, 1])
          intrinsics = make_intrinsics_matrix(tf.multiply(tf.to_float([0.5]), tf.to_float(self.image_width)),
                                              tf.multiply(tf.to_float([0.5]), tf.to_float(self.image_height)),
                                              tf.multiply(tf.to_float([0.5]), tf.to_float(self.image_width)),
                                              tf.multiply(tf.to_float([0.5]), tf.to_float(self.image_height)))

          # Put everything into a dictionary.
          instance = {}
          instance['tgt_image'] = tgt_image
          instance['ref_image'] = ref_image
          instance['src_image'] = src_image

          instance['src_pose'] = pose_two
          instance['ref_pose'] = pose_one
          instance['tgt_pose'] = pose_three

          instance['intrinsics'] = intrinsics
          instance['scene_id'] = scene_id
          instance['image_id'] = image_id
          return instance

    return format_sequence

class RealEstateSequenceDataLoader(object):
  """Loader for video sequence data."""

  def __init__(self,
               cameras_glob='train/????????????????.txt',
               image_dir='images',
               training=True,
               num_source=2,
               shuffle_seq_length=10,
               random_seed=8964,
               map_function=None):

    self.num_source = num_source
    self.random_seed = random_seed
    self.shuffle_seq_length = shuffle_seq_length
    self.batch_size = FLAGS.batch_size
    self.image_height = FLAGS.height
    self.image_width = FLAGS.width

    self.datasets = loader.create_realestate_loader_from_flags(
        cameras_glob=cameras_glob,
        image_dir=image_dir,
        training=training,
        map_function=map_function)

  def set_shapes(self, examples):
    """Set static shapes of the mini-batch of examples.

    Args:
      examples: a batch of examples
    Returns:
      examples with correct static shapes
    """
    b = self.batch_size
    h = self.image_height
    w = self.image_width
    s = self.num_source
    examples['tgt_image'].set_shape([b, h, w, 3])
    examples['ref_image'].set_shape([b, h, w, 3])
    examples['tgt_pose'].set_shape([b, 4, 4])
    examples['ref_pose'].set_shape([b, 4, 4])
    examples['intrinsics'].set_shape([b, 3, 3])

    # modifications for matryodshka training script
    examples['src_pose'].set_shape([b, 4, 4])
    examples['src_image'].set_shape([b, h, w, 3])

    return examples

  def sample_batch(self):
    """Samples a batch of examples for training / testing.

    Returns:
      A batch of examples.
    """
    example = self.datasets.sequences.map(self.format_for_mpi())
    iterator = example.make_one_shot_iterator()
    return self.set_shapes(iterator.get_next())

  def format_for_mpi(self):
    """Format the sampled sequence for MPI training/inference.
    """

    def make_intrinsics_matrix(fx, fy, cx, cy):
      # Assumes batch input.
      batch_size = fx.get_shape().as_list()[0]
      zeros = tf.zeros_like(fx)
      r1 = tf.stack([fx, zeros, cx], axis=1)
      r2 = tf.stack([zeros, fy, cy], axis=1)
      r3 = tf.constant([0., 0., 1.], shape=[1, 3])
      r3 = tf.tile(r3, [batch_size, 1])
      intrinsics = tf.stack([r1, r2, r3], axis=1)
      return intrinsics

    def format_sequence(sequence):
      tgt_idx = tf.random_uniform(
          [],
          maxval=self.shuffle_seq_length,
          dtype=tf.int32,
          seed=self.random_seed)

      shuffled_inds = tf.random_shuffle(tf.range(self.shuffle_seq_length), seed=self.random_seed)
      src_inds = shuffled_inds[:2]
      ref_idx = src_inds[0]
      src_idx = src_inds[1]

      images = sequence.image
      images.set_shape([
          self.batch_size, self.shuffle_seq_length, self.image_height,
          self.image_width, 3
      ])

      poses = sequence.pose
      poses.set_shape([self.batch_size, self.shuffle_seq_length, 3, 4])

      intrinsics = sequence.intrinsics
      intrinsics.set_shape([self.batch_size, self.shuffle_seq_length, 4])

      tgt_image = images[:, tgt_idx]
      ref_image = images[:, ref_idx]
      src_images = images[:, src_idx]

      # Make the pose matrix homogeneous.
      filler = tf.constant(
          [0., 0., 0., 1.], dtype=tf.float32, shape=[1, 1, 1, 4])
      filler = tf.tile(filler, [self.batch_size, self.shuffle_seq_length, 1, 1])
      poses_h = tf.concat([poses, filler], axis=2)
      ref_pose = poses_h[:, ref_idx]
      tgt_pose = poses_h[:, tgt_idx]
      src_poses = poses_h[:, src_idx]

      intrinsics = intrinsics[:, ref_idx]
      intrinsics = make_intrinsics_matrix(intrinsics[:, 0] * self.image_width,
                                          intrinsics[:, 1] * self.image_height,
                                          intrinsics[:, 2] * self.image_width,
                                          intrinsics[:, 3] * self.image_height)
      src_timestamps = tf.gather(sequence.timestamp, src_inds, axis=1)
      ref_timestamp = tf.gather(sequence.timestamp, ref_idx, axis=1)
      tgt_timestamp = tf.gather(sequence.timestamp, tgt_idx, axis=1)

      # Put everything into a dictionary.
      instance = {}
      instance['tgt_image'] = tgt_image
      instance['ref_image'] = ref_image
      instance['src_image'] = src_images
      instance['tgt_pose'] = tgt_pose
      instance['src_pose'] = src_poses
      instance['intrinsics'] = intrinsics
      instance['ref_pose'] = ref_pose
      instance['ref_name'] = sequence.id
      instance['src_timestamps'] = src_timestamps
      instance['ref_timestamp'] = ref_timestamp
      instance['tgt_timestamp'] = tgt_timestamp

      return instance

    return format_sequence
