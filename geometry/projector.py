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
A collection of projection utility functions.
"""
from __future__ import division
import tensorflow as tf
import numpy as np
import homography
import sampling
import spherical
import tensorflow_graphics.geometry.transformation as tfgt

flags = tf.app.flags
FLAGS = flags.FLAGS

def projective_forward_sphere(src_images, intrinsics, tgt_pose_rt, tgt_pos, depths):
    """Project from a position within the sphere sweep volume to an equirect.
    Args:
      src_images: [layers, batch, height, width, channels]
      tgt_pose_rt: [batch, 4, 4] in the form of [R, t]
      tgt_pos: [batch, 3, 1]
      depths: [layers, batch]
    Returns:
      proj_src_images: [layers, batch, height, width, channels]
    """

    n_layers, n_batch, height, width, channels = src_images.get_shape().as_list()

    pixel_coords = []
    for i in range(n_batch):
        pixel_coords_batch = []
        pixels = spherical.intersect_sphere(tgt_pose_rt[i], tgt_pos[i], depths[:, i], n_layers, n_batch, width, height)
        pixel_coords.append(pixels)
    pixel_coords = tf.stack(pixel_coords, axis=0)
    pixel_coords = tf.transpose(pixel_coords,[1,0,2,3,4])

    proj_images = []
    for i in range(n_layers):
        resampled = sampling.bilinear_wrapper2(src_images[i], pixel_coords[i])
        proj_images.append(resampled)

    proj_src_images = tf.stack(proj_images, axis=0)

    return proj_src_images

def projective_forward_sphere_to_perspective(src_images, intrinsics, tgt_pose_rt, tgt_pos, depths, viewing_window = 3, tgt_height=320, tgt_width=640):
    """Project from a position within the sphere sweep volume to a perspective image.
    Args:
      src_images: [layers, batch, height, width, channels]
      tgt_pose_rt: [batch, 4, 4] in the form of [R, t]
      tgt_pos: [batch, 3, 1]
      depths: [layers, batch]
      viewing_direction: 3 default gives the central crop..
    Returns:
      proj_src_images: [layers, batch, height, width, channels]
    """

    n_layers, n_batch, src_height, src_width, channels = src_images.get_shape().as_list()
    pixel_coords = []

    #rotate by 270 degree around y-axis to crop the center view
    angles = [[0., viewing_window * np.pi /2., 0.]]
    rot = tfgt.rotation_matrix_3d.from_euler(angles)
    tr = tf.zeros([1,3,1])
    tgt_pose_rt = tf.concat([rot, tr],axis=2)
    tgt_pose_rt = tf.concat([tgt_pose_rt, tf.expand_dims(tf.eye(4),0)[:,3:,:]],axis=1)
    tgt_pose_rt = tf.tile(tgt_pose_rt,[n_batch,1,1])

    for i in range(n_batch):
        pixel_coords_batch = []
        pixels = spherical.intersect_perspective(tgt_pose_rt[i], tgt_pos[i], depths[:, i], n_layers, n_batch, src_width, src_height, tgt_width, tgt_height, intrinsics)
        pixel_coords.append(pixels)
    pixel_coords = tf.stack(pixel_coords, axis=0)
    pixel_coords = tf.transpose(pixel_coords,[1,0,2,3,4])

    proj_images = []
    for i in range(n_layers):
        resampled = sampling.bilinear_wrapper2(src_images[i], pixel_coords[i])
        proj_images.append(resampled)
    proj_src_images = tf.stack(proj_images, axis=0)
    return proj_src_images

def projective_forward_ods(src_images, order, intrinsics, jitter_pose, pos, depths):
    """Project from a position within the sphere sweep volume to an ods image.

    Args:
      src_images: [layers, batch, height, width, channels]
      pos: [batch, 3]
      depths: [layers, batch]
    Returns:
      proj_src_images: [layers, batch, height, width, channels]
    """
    n_layers, n_batch, height, width, channels = src_images.get_shape().as_list()

    pixel_coords = []
    for i in range(n_batch):
        pixel_coords_batch = []
        pixels = spherical.intersect_ods(jitter_pose[i], pos[i], order, intrinsics, depths[:, i], n_layers, n_batch, width, height)
        pixel_coords.append(pixels)
    pixel_coords = tf.stack(pixel_coords, axis=0)
    pixel_coords = tf.transpose(pixel_coords,[1,0,2,3,4])

    proj_images = []
    for i in range(n_layers):
        resampled = sampling.bilinear_wrapper2(src_images[i], pixel_coords[i])
        proj_images.append(resampled)
    proj_src_images = tf.stack(proj_images, axis=0)

    return proj_src_images

def sweep_one(image, order, depths, pose, intrinsics, st_fun, backproj_fun, proj_fun):

  batch, height, width, channels = image.get_shape().as_list()

  if isinstance(depths, list):
      num_planes = len(depths)
  else:
      num_planes, = depths.get_shape().as_list()

  # Construct S, T
  S, T = st_fun([height, width])

  # Backproject points into reference frame MPI
  all_resampled = []
  for i in range(batch):
      # Pose and intrinsics
      intrinsic = tf.slice(intrinsics,[i,0,0],[1,3,3])
      intrinsic = tf.concat([intrinsic, tf.zeros([1, 1, 3], tf.float32)], axis=1)
      intrinsic = tf.concat([intrinsic, tf.zeros([1, 4, 1], tf.float32)], axis=2)
      intrinsic_tiled = tf.tile(intrinsic, [num_planes, 1, 1])
      pose_one = tf.slice(pose,[i,0,0],[1,4,4])
      pose_tiled = tf.tile(pose_one, [num_planes, 1, 1])

      points = backproj_fun(S, T, tf.convert_to_tensor(depths, tf.float32), intrinsic_tiled)

      # Apply pose
      points = apply_pose(points, pose_tiled)

      # Project points into source frame
      pixel_coords = proj_fun(points, order, pose_tiled, intrinsic_tiled, width, height)

      # Resample
      image_one = tf.slice(image,[i,0,0,0],[1,height,width,channels])
      image_tiled = tf.tile(image_one, [num_planes, 1, 1, 1])
      resampled = sampling.bilinear_wrapper2(image_tiled, pixel_coords)
      resampled = tf.transpose(resampled, [1, 2, 0, 3])

      all_resampled.append(resampled)

  all_resampled = tf.stack(all_resampled)
  resampled = tf.reshape(all_resampled, [batch, height, width, channels * num_planes])
  return resampled

def gcn_sweep_one(image, order, depths, coord, pose, intrinsics, st_fun, backproj_fun, proj_fun):

  batch, height, width, channels = image.get_shape().as_list()
  n_vertex, _ = coord.get_shape().as_list()
  num_planes = len(depths)

  # Backproject points into reference frame MPI
  all_resampled = []
  for i in range(batch):
      # Pose and intrinsics
      intrinsic = tf.slice(intrinsics,[i,0,0],[1,3,3])
      intrinsic = tf.concat([intrinsic, tf.zeros([1, 1, 3], tf.float32)], axis=1)
      intrinsic = tf.concat([intrinsic, tf.zeros([1, 4, 1], tf.float32)], axis=2)
      intrinsic_tiled = tf.tile(intrinsic, [num_planes, 1, 1])

      pose_one = tf.slice(pose,[i,0,0],[1,4,4])
      pose_tiled = tf.tile(pose_one, [num_planes, 1, 1])

      points = coord
      points = tf.expand_dims(tf.transpose(points,[1,0]),axis=0)
      points = expand_along_depth(points, tf.convert_to_tensor(depths, tf.float32))

      # Project points into source frame
      pixel_coords = proj_fun(points, order, pose_tiled, intrinsic_tiled, width, height)

      # Resample
      image_one = tf.slice(image,[i,0,0,0],[1,height,width,channels])
      image_tiled = tf.tile(image_one, [num_planes, 1, 1, 1])
      resampled = sampling.sph_bilinear_wrapper(image_tiled, pixel_coords)
      resampled = tf.transpose(resampled, [1, 2, 0, 3])

      all_resampled.append(resampled)

  all_resampled = tf.stack(all_resampled)
  resampled = tf.reshape(all_resampled, [batch, 1, n_vertex , channels * num_planes])
  return resampled

def ods_sphere_sweep(image, order, depths, pose, intrinsics):
  return sweep_one(image, order, depths, pose, intrinsics,
      spherical.lat_long_grid, spherical.backproject_spherical, spherical.project_ods)

def ods_centered_sphere_sweep(image, order, depths, pose, intrinsics):
  return sweep_one(image, order, depths, pose, intrinsics,
      spherical.lat_long_grid, spherical.backproject_spherical, spherical.project_spherical)

def gcn_sphere_sweep(image, order, depths, coord, pose, intrinsics):
  return gcn_sweep_one(image, order, depths, coord, pose, intrinsics,
      spherical.lat_long_grid, spherical.backproject_spherical, spherical.project_ods)

def perspective_plane_sweep(image, order, depths, pose, intrinsics):
  return sweep_one(image, order, depths, pose, intrinsics,
      spherical.uv_grid, spherical.backproject_planar, spherical.project_perspective)

def over_composite_depth(rgbas):
  """Combines a list of alpha images using the over operation.

  Combines Alpha images from back to front with the over operation.
  The alpha image of the first image is ignored and assumed to be 0.0.

  Args:
    rgbas: A list of [batch, H, W, 4] RGBA images, combined from back to front.
  Returns:
    Composited depth image.
  """

  for i in range(0, len(rgbas)):
    alpha_image = tf.tile(rgbas[i][:, :, :, 3:],[1,1,1,3])
    if i == 0:
      output = 0;
    else:
      output = (i / len(rgbas)) * alpha_image + output * (1.0 - alpha_image)

  return output

def over_composite(rgbas):
  """Combines a list of alpha images using the over operation.

  Combines RGBA images from back to front with the over operation.
  The alpha image of the first image is ignored and assumed to be 1.0.

  Args:
    rgbas: A list of [batch, H, W, 4] RGBA images, combined from back to front.
  Returns:
    Composited RGB image.
  """
  for i in range(len(rgbas)):
    rgb = rgbas[i][:, :, :, 0:3]
    alpha = rgbas[i][:, :, :, 3:]
    if i == 0:
      output = rgb
    else:
      rgb_by_alpha = rgb * alpha
      output = rgb_by_alpha + output * (1.0 - alpha)
  return output

def expand_along_depth(points, depths):
    # points [1, 3, num_vertex]
    num_planes = depths.get_shape().as_list()[0]
    depths = tf.reshape(depths, [num_planes, 1, 1])
    points = tf.tile(points, [num_planes, 1 , 1])
    points = depths * points
    return points

def apply_pose(points, pose):
    x, y, z = points
    batch, h, w = x.get_shape().as_list()
    # print("Applying pose")

    # Homogenous coordinates
    points = tf.stack([x, y, z, tf.ones_like(x)], axis=1)
    points = tf.reshape(points, [batch, 4, -1])

    # Transform by pose
    points = tf.matmul(pose, points)

    # Reshape
    points = tf.reshape(points, [batch, 4, h, w])
    x, y, z, _ = tf.unstack(points, axis=1)

    return x, y, z

def mesh_to_equirect(meshcolors, p2v):
    '''
    Should return a tensor of shape [n_batch, height, width, channels = 67]
    '''
    # print("Mesh colors shape:", meshcolors.get_shape().as_list())
    # print("P2V lookup table shape:", p2v.get_shape().as_list())
    # print("Results shape:", results.get_shape().as_list())
    # return results

    num_vertex, channels = meshcolors.get_shape().as_list() # 0-32 blending weights; 32-64 alpha; 64-67 bg
    width, height, _, _ = p2v.get_shape().as_list()
    image_shape = [1, width, height, channels]

    v1, v2, v3 = tf.unstack(p2v, axis=2)

    v1_coord = tf.cast(tf.reshape(v1[:,:,0], [-1]), 'int32')
    v2_coord = tf.cast(tf.reshape(v2[:,:,0], [-1]), 'int32')
    v3_coord = tf.cast(tf.reshape(v3[:,:,0], [-1]), 'int32')

    #Weights / Diffirentiability Issues
    v1_weight = tf.reshape(v1[:,:,1], [-1])
    v2_weight = tf.reshape(v2[:,:,1], [-1])
    v3_weight = tf.reshape(v3[:,:,1], [-1])

    v1_colors = tf.gather(meshcolors, v1_coord, axis=0)
    v2_colors = tf.gather(meshcolors, v2_coord, axis=0)
    v3_colors = tf.gather(meshcolors, v3_coord, axis=0)

    v1_weight = tf.expand_dims(v1_weight, 1)
    v2_weight = tf.expand_dims(v2_weight, 1)
    v3_weight = tf.expand_dims(v3_weight, 1)

    res = tf.add_n([v1_weight * v1_colors,
                    v2_weight * v2_colors,
                    v3_weight * v3_colors])

    res = tf.reshape(res, image_shape)
    res = tf.transpose(res, [0,2,1,3])

    return res

### from original stereomag repo ###

# Note that there is a subtle bug in how pixel coordinates are treated during
# projection. The projection code assumes pixels are centered at integer
# coordinates. However, this implies that we need to treat the domain of images
# as [-0.5, W-0.5] x [-0.5, H-0.5], whereas we actually use [0, H-1] x [0,
# W-1]. The outcome is that the principal point is shifted by a half-pixel from
# where it should be. We do not believe this issue makes a significant
# difference to the results, however.
def projective_forward_homography(src_images, intrinsics, pose, depths):
  """
  Use homography for forward warping.
  Args:
    src_images: [layers, batch, height, width, channels]
    intrinsics: [batch, 3, 3]
    pose: [batch, 4, 4]
    depths: [layers, batch]
  Returns:
    proj_src_images: [layers, batch, height, width, channels]
  """
  n_layers, n_batch, height, width, _ = src_images.get_shape().as_list()
  # Format for planar_transform code:
  # rot: relative rotation, [..., 3, 3] matrices
  # t: [B, 3, 1], translations from source to target camera (R*p_s + t = p_t)
  # n_hat: [L, B, 1, 3], plane normal w.r.t source camera frame [0,0,1]
  #        in our case
  # a: [L, B, 1, 1], plane equation displacement (n_hat * p_src + a = 0)

  rot = pose[:, :3, :3]
  t = pose[:, :3, 3:]

  n_hat = tf.constant([0., 0., 1.], shape=[1, 1, 1, 3])
  n_hat = tf.tile(n_hat, [n_layers, n_batch, 1, 1])
  a = -tf.reshape(depths, [n_layers, n_batch, 1, 1])
  k_s = intrinsics
  k_t = intrinsics
  pixel_coords_trg = tf.transpose(meshgrid_abs(n_batch, height, width), [0, 2, 3, 1])
  proj_src_images = homography.planar_transform(src_images, pixel_coords_trg, k_s, k_t, rot, t, n_hat, a)

  return proj_src_images

def plane_sweep(image, depths, pose, intrinsics):
  """Construct a plane sweep volume.

  Args:
    image: source image [batch, height, width, #channels]
    depths: a list of depth values for each plane
    pose: target to source camera transformation [batch, 4, 4]
    intrinsics: camera intrinsics [batch, 3, 3]
  Returns:
    A plane sweep volume [batch, height, width, #planes*#channels]
  """
  batch, height, width, _ = image.get_shape().as_list()
  plane_sweep_volume = []

  for depth in depths:
    curr_depth = tf.constant(
        depth, dtype=tf.float32, shape=[batch, height, width])
    warped_image = projective_inverse_warp(image, curr_depth, pose, intrinsics)
    plane_sweep_volume.append(warped_image)
  plane_sweep_volume = tf.concat(plane_sweep_volume, axis=3)
  return plane_sweep_volume

def projective_inverse_warp(img, depth, pose, intrinsics, ret_flows=False):
  """Inverse warp a source image to the target image plane based on projection.

  Args:
    img: the source image [batch, height_s, width_s, 3]
    depth: depth map of the target image [batch, height_t, width_t]
    pose: target to source camera transformation matrix [batch, 4, 4]
    intrinsics: camera intrinsics [batch, 3, 3]
    ret_flows: whether to return the displacements/flows as well
  Returns:
    Source image inverse warped to the target image plane [batch, height_t,
    width_t, 3]
  """
  batch, height, width, _ = img.get_shape().as_list()

  # Construct pixel grid coordinates.
  pixel_coords = meshgrid_abs(batch, height, width)

  # Convert pixel coordinates to the camera frame.
  cam_coords = pixel2cam(depth, pixel_coords, intrinsics)

  # Construct a 4x4 intrinsic matrix.
  filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
  filler = tf.tile(filler, [batch, 1, 1])
  intrinsics = tf.concat([intrinsics, tf.zeros([batch, 3, 1])], axis=2)
  intrinsics = tf.concat([intrinsics, filler], axis=1)

  # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
  # pixel frame.
  proj_tgt_cam_to_src_pixel = tf.matmul(intrinsics, pose)
  src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)

  output_img = sampling.bilinear_wrapper(img, src_pixel_coords)
  if ret_flows:
    return output_img, src_pixel_coords - cam_coords
  else:
    return output_img

def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
  """Transforms coordinates in the pixel frame to the camera frame.

  Args:
    depth: [batch, height, width]
    pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
    intrinsics: camera intrinsics [batch, 3, 3]
    is_homogeneous: return in homogeneous coordinates
  Returns:
    Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
  """
  batch, height, width = depth.get_shape().as_list()
  depth = tf.reshape(depth, [batch, 1, -1])
  pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
  intrinsics_inv = tf.get_default_graph().get_tensor_by_name("intrinsics_inv:0")
  cam_coords = tf.matmul(intrinsics_inv, pixel_coords) * depth

  if is_homogeneous:
    ones = tf.ones([batch, 1, height*width])
    cam_coords = tf.concat([cam_coords, ones], axis=1)

  cam_coords = tf.reshape(cam_coords, [batch, -1, height, width])
  return cam_coords

def cam2pixel(cam_coords, proj):
  """Transforms coordinates in a camera frame to the pixel frame.

  Args:
    cam_coords: [batch, 4, height, width] proj: [batch, 4, 4]
  Returns:
    Pixel coordinates projected from the camera frame [batch, height, width, 2]
  """
  batch, _, height, width = cam_coords.get_shape().as_list()
  cam_coords = tf.reshape(cam_coords, [batch, 4, -1])
  unnormalized_pixel_coords = tf.matmul(proj, cam_coords)

  xy_u = unnormalized_pixel_coords[:, 0:2, :]
  z_u = unnormalized_pixel_coords[:, 2:3, :]

  pixel_coords = xy_u / (z_u + 1e-10)
  pixel_coords = tf.reshape(pixel_coords, [batch, 2, height, width])
  return tf.transpose(pixel_coords, perm=[0, 2, 3, 1])

def meshgrid_abs(batch, height, width, is_homogeneous=True):
  """Construct a 2D meshgrid in the absolute coordinates.

  Args:
    batch: batch size
    height: height of the grid
    width: width of the grid
    is_homogeneous: whether to return in homogeneous coordinates
  Returns:
    x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
  """
  xs = tf.linspace(0.0, tf.cast(width-1, tf.float32), width)
  ys = tf.linspace(0.0, tf.cast(height-1, tf.float32), height)
  xs, ys = tf.meshgrid(xs, ys)

  if is_homogeneous:
    ones = tf.ones_like(xs)
    coords = tf.stack([xs, ys, ones], axis=0)
  else:
    coords = tf.stack([xs, ys], axis=0)
  coords = tf.tile(tf.expand_dims(coords, 0), [batch, 1, 1, 1])
  return coords
