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

"""Collection of utility functions."""

import math
import numpy as np
import PIL.Image as pil
from scipy import signal
import tensorflow as tf
import numpy as np
import numpy.linalg
import tensorflow_graphics.geometry.transformation as tform
import tensorflow_graphics.math.interpolation.slerp as slerp
import cPickle as pickle

flags = tf.app.flags
FLAGS = flags.FLAGS

def load_mesh_input():
	"""
	Load in the pickle file that stores the vertex coords and the support sparse tensors (needed for graph convolution with neighbors)
	support format:
	- A List of size two;
	- Each list element is a tuple: (indices, values, dense_shape) of the sparse tensor

	Please find in Pixel2Mesh repo how the .dat file is created.
	"""

    #load the spherical mesh vertices coordinates and the convolution support matrices
	pkl = pickle.load(open('glob/train/gcn/sphere%d.dat' % FLAGS.subdiv, 'rb'))
	coord = pkl[0]
	support = pkl[1]

    #load the precomputed pixel-to-vertex lookup dictionary
	pix2vertex = np.load('glob/train/gcn/p2v%d.npy' % FLAGS.subdiv)
	return coord, support, pix2vertex

def interpolate_pose(ref_pose, src_pose):
    # Interpolate rotation
    ref_rot = ref_pose[:, :3, :3]
    src_rot = src_pose[:, :3, :3]

    ref_quat = tform.quaternion.from_rotation_matrix(ref_rot)
    src_quat = tform.quaternion.from_rotation_matrix(src_rot)

    out_quat = slerp.interpolate(ref_quat, src_quat, 0.5)
    out_rot = tform.rotation_matrix_3d.from_quaternion(out_quat)

    # Interpolate translation
    ref_t = ref_pose[:, :3, 3:]
    src_t = src_pose[:, :3, 3:]
    out_t = 0.5 * ref_t + 0.5 * src_t

    # Output
    combined = tf.concat([out_rot, out_t], axis=2)

    return tf.concat([combined, ref_pose[:, 3:, :]], axis=1)

def write_image(filename, image):
  """Save image to disk."""
  byte_image = np.clip(image, 0, 255).astype('uint8')
  image_pil = pil.fromarray(byte_image)
  with tf.gfile.GFile(filename, 'w') as fh:
    image_pil.save(fh)

def write_pose(filename, pose):
  with tf.gfile.GFile(filename, 'w') as fh:
    for i in range(3):
      for j in range(4):
        fh.write('%f ' % (pose[i, j]))


def write_intrinsics(fh, intrinsics):
  fh.write('%f ' % intrinsics[0, 0])
  fh.write('%f ' % intrinsics[1, 1])
  fh.write('%f ' % intrinsics[0, 2])
  fh.write('%f ' % intrinsics[1, 2])


def build_matrix(elements):
  """Stacks elements along two axes to make a tensor of matrices.
  Args:
    elements: [n, m] matrix of tensors, each with shape [...].

  Returns:
    [..., n, m] tensor of matrices, resulting from concatenating
      the individual tensors.
  """
  rows = [tf.stack(row_elements, axis=-1) for row_elements in elements]
  return tf.stack(rows, axis=-2)

def build_inv_matrix(elements):
  rows = [np.stack(row_elements, axis=-1) for row_elements in elements]
  mat = np.stack(rows, axis=-2)
  mat = mat.astype(np.float32)
  inv_mat = np.linalg.inv(mat)
  return tf.convert_to_tensor(inv_mat, dtype=tf.float32)
