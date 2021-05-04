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

"""Loads video sequence data using the tf Dataset API."""
import collections
import os.path
import tensorflow as tf
import utils
from tensorflow import flags
FLAGS = flags.FLAGS

class OdsSequence(
    collections.namedtuple('OdsSequence',
                           ['scene_id', 'image_id', 'baseline', 'tgt_pos', 'image', 'hres_image', 'pose_inv'])):
  """
    Attributes:
      scene_id: [] string id for identifying scene and positions.
      image_id: [3, 1] image id for src, ref, tgt images
      tgt_pos: [3, 1] translation vector for target image
      image: [n, height, width, channels] image data tensor
      hres_image: [n, hres_height, hres_width, channels] highres image data tensor if highres flag is set
  """
  def length(self):
      """Needed because len returns the number of attributes, not their length."""
      return tf.shape(self.image_id)[0]

  def set_batched_shape(self, batch_size, sequence_length):
    """Set the shape of the sequence to reflect batch size and sequence length.

    Only call this when you know that sequences are of this length and in
    batches of this size and you just want to make sure the static shape
    reflects that.

    Args:
      batch_size: (int) the batch size
      sequence_length: (int) the sequence length

    Returns:
      the input sequence.
    """

    def batch_one(tensor, dims, match_sequence_length=True):
      """Set shape for one tensor."""
      shape = tensor.get_shape().as_list()
      assert len(shape) == dims
      if shape[0] is None:
        shape[0] = batch_size
      else:
        assert shape[0] == batch_size
      if dims >= 2 and match_sequence_length:
        if shape[1] is None:
          shape[1] = sequence_length
        else:
          assert shape[1] == sequence_length
      tensor.set_shape(shape)

    batch_one(self.scene_id, 1)
    batch_one(self.image_id, 2)
    batch_one(self.baseline, 1)
    batch_one(self.image, 5)
    if 'hrestgt' in FLAGS.supervision:
        batch_one(self.hres_image, 5)

    return self

class ReplicaPerspectiveSequence(
    collections.namedtuple('ReplicaPerspectiveSequence',
                           ['scene_id', 'image_id', 'input_offset', 'tgt_offset', 'image'])):
  """A simple wrapper for ViewSequence tensors.

    Attributes:
      scene_id: [] string id for identifying scene and positions.
      image_id: [3, 1] image id for src, ref, tgt images
      input_offset: [3, 1] translation vector for target image
      tgt_offset:
      image: [4, height, width, channels] image data tensor.

  (If data is subsequently batched there may be an additional batch
  dimension at the start of all the above shapes.)
  """

  def length(self):
    """Needed because len returns the number of attributes, not their length."""
    return tf.shape(self.image_id)[0]

  def set_batched_shape(self, batch_size, sequence_length):
    """Set the shape of the sequence to reflect batch size and sequence length.

    Only call this when you know that sequences are of this length and in
    batches of this size and you just want to make sure the static shape
    reflects that.

    Args:
      batch_size: (int) the batch size
      sequence_length: (int) the sequence length

    Returns:
      the input sequence.
    """

    def batch_one(tensor, dims, match_sequence_length=True):
      """Set shape for one tensor."""
      shape = tensor.get_shape().as_list()
      assert len(shape) == dims
      if shape[0] is None:
        shape[0] = batch_size
      else:
        assert shape[0] == batch_size
      if dims >= 2 and match_sequence_length:
        if shape[1] is None:
          shape[1] = sequence_length
        else:
          assert shape[1] == sequence_length
      tensor.set_shape(shape)

    batch_one(self.scene_id, 1)
    batch_one(self.image_id, 2)
    batch_one(self.input_offset, 1)
    batch_one(self.tgt_offset, 1)
    batch_one(self.image, 5)
    return self

class RealEstateViewSequence(
    collections.namedtuple('RealEstateViewSequence',
                           ['id', 'timestamp', 'intrinsics', 'pose', 'image'])):
  """A simple wrapper for ViewSequence tensors.

  A ViewSequence stores camera intrinsics, poses, timestamps and (optionally)
  image data for a series of N views.

  Attributes:
    id: [] string id for train/test split hashing. E.g. video URL.
    timestamps: [N] strings
    intrinsics: [N, 4] (fx fy cx cy) in normalised image coordinates
    pose: [N, 3, 4] rigid transformation matrix mapping world to camera space
    image: [N, height, width, channels] image data tensor, or [N] 0.0 if this
      sequence doesn't have images. (We can't use None because the Dataset API
      requires everything to be a tensor.)

  (If data is subsequently batched there may be an additional batch
  dimension at the start of all the above shapes.)
  """

  def set_batched_shape(self, batch_size, sequence_length):
    """Set the shape of the sequence to reflect batch size and sequence length.

    Only call this when you know that sequences are of this length and in
    batches of this size and you just want to make sure the static shape
    reflects that.

    Args:
      batch_size: (int) the batch size
      sequence_length: (int) the sequence length

    Returns:
      the input sequence.
    """

    def batch_one(tensor, dims, match_sequence_length=True):
      """Set shape for one tensor."""
      shape = tensor.get_shape().as_list()
      assert len(shape) == dims
      if shape[0] is None:
        shape[0] = batch_size
      else:
        assert shape[0] == batch_size
      if dims >= 2 and match_sequence_length:
        if shape[1] is None:
          shape[1] = sequence_length
        else:
          assert shape[1] == sequence_length
      tensor.set_shape(shape)

    batch_one(self.id, 1)
    batch_one(self.timestamp, 2)
    batch_one(self.intrinsics, 3)
    batch_one(self.pose, 4)
    batch_one(self.image, 5)
    return self

  def length(self):
    """Needed because len returns the number of attributes, not their length."""
    return tf.shape(self.pose)[0]

  def subsequence(self, start, end, stride=1):
    """Extracts a subsequence, with [start:end:stride] on each attribute."""
    # Fail rather than return a subsquence that isn't the length the caller
    # would be expecting.
    with tf.control_dependencies([
        tf.Assert(
            tf.logical_and(
                tf.greater_equal(start, 0), tf.less_equal(end, self.length())),
            [start, end, self.length()])
    ]):
      pose = self.pose[start:end:stride]
      intrinsics = self.intrinsics[start:end:stride]
      timestamp = self.timestamp[start:end:stride]
      image = self.image[start:end:stride]
      # Strided subsequences cause TF to forget the shapes, so we have to set
      # them again.
      timestamp.set_shape([None])
      pose.set_shape([None, 3, 4])
      intrinsics.set_shape([None, 4])
      # image could be [N, H, W, C], or it could be [N], so we read the old
      # shape to set the new one.
      image.set_shape([None] + self.image.get_shape().as_list()[1:])

    return RealEstateViewSequence(self.id, timestamp, intrinsics, pose, image)

  def central_subsequence(self, sequence_length, stride):
    """Extracts a centred subsequence with this length and stride, or fails."""
    total_length = (sequence_length - 1) * stride + 1
    index = (self.length() - total_length) // 2
    return self.subsequence(index, index + total_length, stride=stride)

  def reverse(self):
    return RealEstateViewSequence(self.id, tf.reverse(self.timestamp, [0]),
                                      tf.reverse(self.intrinsics, [0]),
                                      tf.reverse(self.pose, [0]), tf.reverse(self.image, [0]))

  def random_subsequence(self, length, min_stride=1, max_stride=1):
    """Returns a random subsequence with min_stride <= stride <= max_stride.

    For example if self.length = 4 and we ask for a length 2
    sequence (with default min/max_stride=1), there are three possibilities:
    [0,1], [1,2], [2,3].

    Args:
      length: the length of the subsequence to be returned.
      min_stride: the minimum stride (> 0) between elements of the sequence
      max_stride: the maximum stride (> 0) between elements of the sequence

    Returns:
      A random, uniformly chosen subsequence of the requested length
      and stride.
    """
    # First pick a stride.
    if max_stride == min_stride:
      stride = min_stride
    else:
      stride = tf.random_uniform(
          [], minval=min_stride, maxval=max_stride + 1, dtype=tf.int32)

    # Now pick the starting index.
    # If the subsequence starts at index i, then its final element will be at
    # index i + (length - 1) * stride, which must be less than the length of
    # the sequence. Therefore i must be less than maxval, where:
    maxval = self.length() - (length - 1) * stride
    index = tf.random_uniform([], maxval=maxval, dtype=tf.int32)
    return self.subsequence(
        index, index + 1 + (length - 1) * stride, stride=stride)

  def random_reverse(self):
    """Returns either the sequence or its reverse, with equal probability."""
    uniform_random = tf.random_uniform([], 0, 1.0)
    condition = tf.less(uniform_random, 0.5)
    return tf.cond(condition, lambda: self, lambda: self.reverse())  # pylint: disable=unnecessary-lambda

  def deterministic_reverse(self):
    """Returns either the sequence or its reverse, based on the sequence id."""
    return tf.cond(
        self.hash_in_range(2, 0, 1), lambda: self, lambda: self.reverse())  # pylint: disable=unnecessary-lambda

  def random_scale_and_crop(self, min_scale, max_scale, height, width):
    """Randomly scale and crop sequence, for data augmentation.

    Args:
      min_scale: (float) minimum scale factor
      max_scale: (float) maximum scale factor
      height: (int) height of output images
      width: (int) width of output images

    Returns:
      A version of this sequence in which all images have been scaled in x and y
      by factors randomly chosen from min_scale to max_scale, and randomly
      cropped to give output images of the requested dimensions. Scaling and
      cropping are done consistently for all images in the sequence, and
      intrinsics are adjusted accordingly.
    """
    if min_scale == 1.0 and max_scale == 1.0:
      scaled_image = self.image
    else:
      input_size = tf.to_float(tf.shape(self.image)[-3:-1])
      scale_factor = tf.random_uniform([2], min_scale, max_scale)
      scaled_image = tf.image.resize_area(
          self.image, tf.to_int32(input_size * scale_factor))

    # Choose crop offset
    scaled_size = tf.shape(scaled_image)[-3:-1]
    offset_limit = scaled_size - [height, width] + 1
    offset_y = tf.random_uniform([], 0, offset_limit[0], dtype=tf.int32)
    offset_x = tf.random_uniform([], 0, offset_limit[1], dtype=tf.int32)

    image, intrinsics = crop_image_and_adjust_intrinsics(
        scaled_image, self.intrinsics, offset_y, offset_x, height, width)
    return RealEstateViewSequence(self.id, self.timestamp, intrinsics, self.pose, image)

  def hash_in_range(self, buckets, base, limit):
    """Return true if the hashed id falls in the range [base, limit)."""
    hash_bucket = tf.string_to_hash_bucket_fast(self.id, buckets)
    return tf.logical_and(
        tf.greater_equal(hash_bucket, base), tf.less(hash_bucket, limit))

def read_file_lines(filename, max_lines=10000):
  """Reads a text file, skips comments, and lines.

  Args:
    filename: the file to read
    max_lines: how many lines to combine into a single element. Any
      further lines will be skipped. The Dataset API's batch function
      requires this parameter, which is the batch size, so we set it to
      something bigger than our sequences.

  Returns:
    The lines of the file, as a 1-dimensional string tensor.
  """
  lines = (
      tf.data.TextLineDataset(filename)
      .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), '#'))
      .batch(max_lines).take(1))
  return tf.contrib.data.get_single_element(lines)

def parse_camera_lines(lines):
  """
  Reads a camera file, returning a single ViewSequence (without images).
  Args:
    lines: [N] string tensor of camera lines

  Returns:
    The corresponding length N sequence, as a ViewSequence.
  """
  # The first line contains the YouTube video URL.
  # Format of each subsequent line: timestamp fx fy px py k1 k2 row0 row1  row2
  # Column number:                  0         1  2  3  4  5  6  7-10 11-14 15-18
  youtube_url = lines[0]
  record_defaults = ([['']] + [[0.0]] * 18)
  data = tf.decode_csv(lines[1:], record_defaults, field_delim=' ')

  with tf.control_dependencies([
      # We don't accept non-zero k1 and k2.
      tf.assert_equal(data[5:6], 0.0)
  ]):
    timestamps = data[0]
    intrinsics = tf.stack(data[1:5], axis=1)
    poses = utils.build_matrix([data[7:11], data[11:15], data[15:19]])

  # No image data yet. Ideally we'd put "None" for image, but the dataset
  # API doesn't allow that, so we use zeros instead.
  images = tf.zeros_like(timestamps, dtype=tf.float32)

  # In camera files, the video id is the last part of the YouTube URL, it comes
  # after the =. It seems hacky to use decode_csv but it's easier than
  # string_split because that returns a sparse tensor.
  youtube_id = tf.decode_csv([youtube_url], [[''], ['']], field_delim='=')[1][0]
  return RealEstateViewSequence(youtube_id, timestamps, intrinsics, poses, images)

def load_image_data(base_path, height, width, parallel_image_reads):
  """Returns a mapper function for loading image data.

  Args:
    base_path: (string) The base directory for images
    height: (int) Images will be resized to this height
    width: (int) Images will be resized to this width
    parallel_image_reads: (int) How many images to read in parallel

  Returns:
    A function mapping ViewSequence to ViewSequence, suitable for
    use with map(). The returned ViewSequence will be identical to the
    input one, except that sequence.images have been filled in.
  """
  # Ensure base_path has just one trailing '/'.
  base_path = os.path.dirname(os.path.join(base_path, 'file')) + '/'

  def load_single_image(filename):
    """Load and size a single image from a given filename."""
    contents = tf.read_file(base_path + '/' + filename)
    image = tf.image.convert_image_dtype(
        tf.image.decode_image(contents), tf.float32)
    # Unfortunately resize_area expects batched images, so add a dimension,
    # resize, and then remove it again.
    resized = tf.squeeze(
        tf.image.resize_area(tf.expand_dims(image, axis=0), [height, width]),
        axis=0)
    resized.set_shape([height, width, 3])  # RGB images have 3 channels.
    return resized

  def mapper(sequence):
    images = tf.contrib.data.get_single_element(
        tf.data.Dataset.from_tensor_slices(sequence.id + '/' + sequence.id +
                                           '_' + sequence.timestamp + '.jpg')
        .map(load_single_image, num_parallel_calls=parallel_image_reads).batch(
            tf.to_int64(sequence.length())))
    return RealEstateViewSequence(sequence.id, sequence.timestamp, sequence.intrinsics,
                                  sequence.pose, images)
  return mapper

def parse_replica_ods_camera_lines(lines):
    num_img = FLAGS.shuffle_seq_length
    record_defaults=([['']] + [['']]*num_img + [[0.0]]*20)
    data = tf.decode_csv(lines, record_defaults, field_delim=' ')
    scene_id = data[0]
    img_id = data[1:1+num_img]
    baseline = data[1+num_img]
    tgt_pos = data[2+num_img:2+num_img+3]
    pose_inv = data[2+num_img+3:24]
    temp = pose_inv[4:8]

    for i in range(len(temp)):
      temp[i] *= -1
    pose_inv[4:8] = pose_inv[8:12]
    pose_inv[8:12] = temp[:]
    pose_inv = tf.reshape(pose_inv, [4, 4])
    pose_inv = tf.expand_dims(pose_inv, axis=0)

    images = tf.zeros_like(img_id, dtype=tf.float32)
    hres_images = tf.zeros_like(img_id, dtype=tf.float32)

    return OdsSequence(scene_id, img_id, baseline, tgt_pos, images, hres_images, pose_inv)

def parse_replica_perspective_camera_lines(lines):
    record_defaults=([['']] + [['']]*3 + [[0.0]]*2)
    data = tf.decode_csv(lines, record_defaults, field_delim=' ')
    scene_id = data[0]
    img_id = data[1:4]
    input_offset = data[4]
    tgt_offset = data[5]

    images = tf.zeros_like(img_id, dtype=tf.float32)

    return ReplicaPerspectiveSequence(scene_id, img_id, input_offset, tgt_offset, images)

def load_replica_perspective_image_data(base_path, height, width, parallel_image_reads):
  """Returns a mapper function for loading equirect image data.

  Args:
    base_path: (string) The base directory for images
    height: (int) Images will be resized to this height
    width: (int) Images will be resized to this width
    parallel_image_reads: (int) How many images to read in parallel

  Returns:
    A function mapping EquirectSequence to EquirectSequence, suitable for
    use with map(). The returned EquirectSequence will be identical to the
    input one, except that sequence.image have been filled in.
  """
  # Ensure base_path has just one trailing '/'.
  # base_path = os.path.dirname(os.path.join(base_path, 'file')) + '/'
  base_path = os.path.dirname(os.path.join(base_path, 'file'))

  def load_single_image(filename):
    """Load and size a single image from a given filename."""
    contents = tf.read_file(base_path + '/' + filename)
    # image = tf.image.convert_image_dtype(
    #     tf.image.decode_png(contents,channels=4), tf.uint8)
    image = tf.image.convert_image_dtype(
        tf.image.decode_jpeg(contents), tf.uint8)

    image = tf.image.convert_image_dtype(image, tf.float32)
    # Unfortunately resize_area expects batched images, so add a dimension,
    # resize, and then remove it again.
    resized = tf.squeeze(
        tf.image.resize_area(tf.expand_dims(image, axis=0), [height, width]),
        axis=0)
    resized = resized[ :, :, :3]
    resized.set_shape([height, width, 3])  # RGB images have 3 channels.
    return resized

  def mapper(sequence):
    images = tf.contrib.data.get_single_element(
        tf.data.Dataset.from_tensor_slices(sequence.scene_id + '_pos' + sequence.image_id + '.jpeg')
        .map(load_single_image, num_parallel_calls=parallel_image_reads).batch(
            tf.to_int64(sequence.length())))

    return ReplicaPerspectiveSequence(sequence.scene_id, sequence.image_id, sequence.input_offset, sequence.tgt_offset, images)

  return mapper

def load_replica_ods_image_data(base_path, hres_base_path, height, width, hres_height, hres_width, parallel_image_reads):
  """Returns a mapper function for loading equirect image data.

  Args:
    base_path: (string) The base directory for images
    height: (int) Images will be resized to this height
    width: (int) Images will be resized to this width
    parallel_image_reads: (int) How many images to read in parallel

  Returns:
    A function mapping EquirectSequence to EquirectSequence, suitable for
    use with map(). The returned EquirectSequence will be identical to the
    input one, except that sequence.image have been filled in.
  """
  # Ensure base_path has just one trailing '/'.
  # base_path = os.path.dirname(os.path.join(base_path, 'file')) + '/'

  #hard code the high res path for now
  # hres_base_path = '/media/selenaling/Elements/train_4096x2048'
  base_path = os.path.dirname(os.path.join(base_path, 'file'))
  hres_base_path = os.path.dirname(os.path.join(hres_base_path, 'file'))

  def load_single_image(filename):
    """Load and size a single image from a given filename."""

    contents = tf.read_file(base_path + '/' + filename)
    image = tf.image.convert_image_dtype(tf.image.decode_jpeg(contents), tf.uint8)
    image = tf.image.convert_image_dtype(image, tf.float32)
    resized = tf.squeeze(
        tf.image.resize_area(tf.expand_dims(image, axis=0), [height, width]),
        axis=0)
    resized = resized[ :, :, :3]
    resized.set_shape([height, width, 3])  # RGB images have 3 channels.

    return resized

  def load_single_highres_image(filename):
    """Load and size a single image from a given filename."""

    contents = tf.read_file(hres_base_path + '/' + filename)
    image = tf.image.convert_image_dtype(tf.image.decode_jpeg(contents), tf.uint8)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Unfortunately resize_area expects batched images, so add a dimension,
    # resize, and then remove it again.
    resized = tf.squeeze(
        tf.image.resize_area(tf.expand_dims(image, axis=0), [hres_height, hres_width]),
        axis=0)
    resized = resized[ :, :, :3]
    resized.set_shape([hres_height, hres_width, 3])  # RGB images have 3 channels.

    return resized

  def mapper(sequence):
    images = tf.contrib.data.get_single_element(
        tf.data.Dataset.from_tensor_slices(sequence.scene_id + '_pos' + sequence.image_id + '.jpeg')
        .map(load_single_image, num_parallel_calls=parallel_image_reads).batch(
            tf.to_int64(sequence.length())))

    hres_images = sequence.hres_image
    if 'hrestgt' in FLAGS.supervision:
        hres_images = tf.contrib.data.get_single_element(
            tf.data.Dataset.from_tensor_slices(sequence.scene_id + '_pos' + sequence.image_id + '.jpeg')
            .map(load_single_highres_image, num_parallel_calls=parallel_image_reads).batch(
                tf.to_int64(sequence.length())))
    return OdsSequence(sequence.scene_id, sequence.image_id, sequence.baseline, sequence.tgt_pos, images, hres_images, sequence.pose_inv)

  return mapper

def crop_image_and_adjust_intrinsics(
    image, intrinsics, offset_y, offset_x, height, width):
  """Crop images and adjust instrinsics accordingly.

  Args:
    image: [..., H, W, C] images
    intrinsics: [..., 4] normalised camera intrinsics
    offset_y: y-offset in pixels from top of image
    offset_x: x-offset in pixels from left of image
    height: height of region to be cropped
    width: width of region to be cropped

  Returns:
    [..., height, width, C] cropped images,
    [..., 4] adjusted intrinsics
  """
  shape = tf.to_float(tf.shape(image))
  original_height = shape[-3]
  original_width = shape[-2]

  # intrinsics = [fx fy cx cy]
  # Convert to pixels, offset, and normalise to cropped size.
  pixel_intrinsics = intrinsics * tf.stack([
      original_width, original_height, original_width, original_height])
  cropped_pixel_intrinsics = (
      pixel_intrinsics - tf.stack(
          [0.0, 0.0, tf.to_float(offset_x), tf.to_float(offset_y)]))
  cropped_intrinsics = (
      cropped_pixel_intrinsics
      / tf.to_float(tf.stack([width, height, width, height])))
  cropped_images = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, height, width)
  return cropped_images, cropped_intrinsics
