#!/usr/bin/python
#
# Copyright 2020 Brown Visual Computing Lab / Authors of the accompanying paper Matryodshka #
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

import numpy as np
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfgt

def tf_random_rotation(rc, tc, angle_range=[-0.03, 0.03], offset_range=[-0.01, 0.01]):
    '''
    rc: coefficient for rotation range
    tc: coefficient for translation range
    angle_range: in radians
    '''

    angle_range = [x*rc for x in angle_range]
    offset_range = [x*tc for x in offset_range]

    # Rotation
    angles = tf.random.uniform([1,3],angle_range[0], angle_range[1])
    rot = tfgt.rotation_matrix_3d.from_euler(angles)

    tr = tf.random.uniform([1,3,1],offset_range[0],offset_range[1])

    mat = tf.concat([rot, tr],axis=2)
    mat = tf.concat([mat, tf.expand_dims(tf.eye(4),0)[:,3:,:]],axis=1)

    return mat

def lat_long_grid(shape, epsilon = 1.0e-12):
    return tf.meshgrid(tf.linspace(-np.pi + np.pi / shape[1], np.pi - np.pi / shape[1], shape[1]),
                       tf.linspace(-np.pi / 2.0 + np.pi / (2 * shape[0]), np.pi / 2.0 - np.pi / (2 * shape[0]), shape[0]))

def uv_grid(shape):
    return tf.meshgrid(tf.linspace(-1. + 1. / shape[1], 1. - 1. / shape[1], shape[1]),
                       tf.linspace(-1. + 1. / shape[0], 1. - 1. / shape[0], shape[0]))

def theta_y_grid(shape):
    return tf.meshgrid(tf.linspace(-np.pi, np.pi, shape[1]),
                       tf.linspace(-1., 1., shape[0]))

def theta_phi_to_pixels(theta, phi, width, height):
    # Get pixel coords as indices, e.g., integers from 0 to width
    # 1. Map to range 0 to 1
    u = (theta + np.pi) # [2pi,0]
    # 2. Subtract on half a pixel to put us in the center of the pixel
    u = u - (np.pi / width)
    # 3. Divide by the width of the space in pixels, which is 2pi - one pixel's width
    u = u / (2 * np.pi - (2 * np.pi / width))
    # 4. Multiply by width
    u = u * (width - 1)
    # Do the same for v
    v = (phi + (0.5 * np.pi) - (0.5 * np.pi / height)) / (np.pi - np.pi / height)
    v = v * (height - 1)
    uv = tf.stack([u, v], axis=-1)
    return uv

def transform_ray(r, c, pos):
    rx, ry, rz = r
    cx, cy, cz = c

    # Transform the ray direction
    ray = tf.stack([rx, ry, rz], axis=0)
    orig_shape = ray.get_shape().as_list()
    ray = tf.reshape(ray, [3, -1])
    rot = pos[:3,:3]
    rot_ray = tf.reshape(tf.matmul(rot, ray), orig_shape) #[3, 32, h, w]
    rx = rot_ray[0]
    ry = rot_ray[1]
    rz = rot_ray[2]

    # Transform ray center
    point = tf.stack([cx, cy, cz, tf.ones_like(cx)], axis=0)
    orig_shape = point.get_shape().as_list()
    point = tf.reshape(point, [4, -1])
    rot_point = tf.reshape(tf.matmul(pos, point), orig_shape)
    cx = rot_point[0]
    cy = rot_point[1]
    cz = rot_point[2]

    return (rx, ry, rz), (cx, cy, cz)

def get_sphere_intersections(r, c, radius):
    rx, ry, rz = r
    cx, cy, cz = c

    # Solve for ray intersection
    a = rx * rx + ry * ry + rz * rz
    b = 2 * (rx * cx + ry * cy + rz * cz)
    c = cx * cx + cy * cy + cz * cz - radius * radius
    disc = tf.square(b) - 4 * a * c

    t = (-b + tf.sqrt(disc)) / (2 * a) # This should be the only solution
    x = cx + t * rx
    y = cy + t * ry
    z = cz + t * rz

    return (x, y, z)

# Backprojection method -- what pixel coords in reference frame correspond to in 3D space
# Needs to correspond to real geometry of setup and camera capture [replica renderer]
# Also defines parametrization for spherical MPI (what pixel corresponds to which point in 3D space)
# Needs to correspond to where placed in final rendering [Unity]
def backproject_spherical(S, T, depth, intrinsics):
    num_planes = depth.get_shape().as_list()[0]

    # Tile
    S = tf.tile(tf.expand_dims(S, 0), [num_planes, 1, 1])
    T = tf.tile(tf.expand_dims(T, 0), [num_planes, 1, 1])
    depth = tf.reshape(depth, [num_planes, 1, 1])

    # Back project
    cosT = tf.cos(T)
    x = depth * (tf.cos(S) * cosT)
    y = depth * tf.sin(T)
    z = -depth * (tf.sin(S) * cosT) #Q: why negation here? does it make a difference? test it out
    return x, y, z

def backproject_planar(S, T, depth, intrinsics):
    num_planes = depth.get_shape().as_list()[0]

    # Tile
    S = tf.tile(tf.expand_dims(S, 0), [num_planes, 1, 1])
    T = tf.tile(tf.expand_dims(T, 0), [num_planes, 1, 1])
    depth = tf.reshape(depth, [num_planes, 1, 1])

    # Get intrinsics
    fx = intrinsics[0, 0, 0]
    fy = intrinsics[0, 1, 1]
    cx = intrinsics[0, 0, 2]
    cy = intrinsics[0, 1, 2]

    # Back project
    x = depth * S * cx / fx
    y = depth * T * cy / fy
    z = depth * tf.ones_like(x)
    return x, y, z

def backproject_cylindrical(S, T, depth, intrinsics):
    num_planes = depth.get_shape().as_list()[0]

    S = tf.tile(tf.expand_dims(S, 0), [num_planes, 1, 1])
    T = tf.tile(tf.expand_dims(T, 0), [num_planes, 1, 1])
    depth = tf.reshape(depth, [num_planes, 1, 1])

    # Get intrinsics
    fy = intrinsics[0, 1, 1]
    cy = intrinsics[0, 1, 2]

    # Back project
    x = depth * tf.cos(S)
    y = depth * T * cy / fy
    z = depth * tf.sin(S)
    return x, y, z

# Projection method -- what uv 3D points correspond to in frame of src image
# Needs to correspond to real geometry of setup and camera capture [replica renderer]
def project_ods(points, order, pose, intrinsics, width, height):

    if tf.is_tensor(points):
        x = points[:,:1,:]
        y = - points[:,1:2,:]
        z = points[:,2:,:]
    else:
        x, y, z = points

    num_planes, _, _ = x.get_shape().as_list()

    r = intrinsics[0][0][0]
    f = r * r - (tf.square(x) + tf.square(z))
    z_larger_x = tf.greater(tf.abs(z), tf.abs(x))
    px = tf.where(z_larger_x, x, z)
    pz = tf.where(z_larger_x, z, x)

    # Solve quadratic
    pz_square = tf.square(pz)
    a = 1 + tf.square(px) / pz_square
    b = -2 * f * px / pz_square
    c = f + tf.square(f) / pz_square
    disc = tf.square(b) - 4 * a * c

    # Direction vector from point
    s = -order * tf.sign(pz) * tf.sqrt(disc)
    s = tf.where(z_larger_x, s, -s)

    dx = (-b + s) / (2 * a)
    dz = (f - px * dx) / pz

    dx_final = tf.where(z_larger_x, -dx, -dz)
    dz_final = tf.where(z_larger_x, -dz, -dx)
    dx = dx_final
    dz = dz_final
    dy = y

    # Angles from direction vector
    theta = -tf.atan2(dz, dx)
    phi = tf.atan2(dy, tf.sqrt(tf.square(dx) + tf.square(dz)))
    nan_mask = tf.is_nan(phi)
    phi = tf.where(nan_mask, tf.ones_like(phi), phi)

    pos_phi = tf.ones_like(dx) * np.pi/2
    neg_phi = tf.ones_like(dx) * np.pi/2 * -1.

    pos_phi_mask = tf.less_equal(phi, np.pi/2)
    neg_phi_mask = tf.greater_equal(phi, -np.pi/2)
    phi = tf.where(pos_phi_mask, phi, pos_phi)
    phi = tf.where(neg_phi_mask, phi, neg_phi)

    # Get pixel coords
    u = ((theta + np.pi - np.pi / width) / (2 * np.pi - 2 * np.pi / width)) * (width - 1)
    v = ((phi + 0.5 * np.pi - 0.5 * np.pi / height) / (np.pi - np.pi / height)) * (height - 1)

    # Keep valid parts
    valid_mask = tf.greater_equal(disc, 0.)
    ones = tf.ones_like(u)
    u = tf.where(valid_mask, u, ones)
    v = tf.where(valid_mask, v, ones)

    # Return
    uv = tf.stack([u, v], axis=-1)
    return uv

def project_spherical(points, order, pose, intrinsics, width, height):
    # TODO: pixel shift for converting to uv
    # TODO: wrap convolution
    # TODO: correct pixel shifts on windows
    x, y, z = points
    num_planes, _, _ = x.get_shape().as_list()

    # Angles from point
    theta = -tf.atan2(z, x) # [-pi,pi], negated, so [pi,-pi]
    phi = tf.atan2(y, tf.sqrt(tf.square(x) + tf.square(z))) # [-pi,pi]

    return theta_phi_to_pixels(theta, phi, width, height)

def project_perspective(points, order, pose, intrinsics, width, height):
    # Extract x, y, z
    x, y, z = points
    batch, _, _ = x.get_shape().as_list()

    # Homogenous coordinates
    points = tf.stack([x, y, z, tf.ones_like(x)], axis=1)
    points = tf.reshape(points, [batch, 4, -1])

    # Transform by pose
    intrinsics_pose = tf.matmul(intrinsics, pose)
    points = tf.matmul(intrinsics_pose, points)

    # Transform to pixel coords
    uv = points[:, 0:2, :] / points[:, 2:3, :]
    uv = tf.transpose(uv, [0, 2, 1])
    uv = tf.reshape(uv, [batch, height, width, 2])

    return uv

def intersect_sphere(pos, center, radius, num_planes, num_batch, width, height, epsilon=1e-12):

    # Pixels in the target image
    S, T = lat_long_grid((height, width))
    S = tf.tile(tf.expand_dims(S, 0), [num_planes, 1, 1])
    T = tf.tile(tf.expand_dims(T, 0), [num_planes, 1, 1])

    # Reshape radius
    radius = tf.reshape(radius, [num_planes, 1, 1])

    # Ray direction
    cosT = tf.cos(T)
    rx = tf.cos(S) * cosT
    ry = tf.sin(T)
    rz = -tf.sin(S) * cosT
    n_layers, h , w = rx.get_shape().as_list()

    # ray center
    cx = center[0]
    cy = center[1]
    cz = -center[2]

    # Transform the ray/center with pose
    ray = tf.stack([rx, ry, rz], axis=0)
    orig_shape = ray.get_shape().as_list()
    ray = tf.reshape(ray, [3, -1])
    rot = pos[:3,:3]
    rot_ray = tf.reshape(tf.matmul(rot, ray), orig_shape) #[3, 32, h, w]
    rx = rot_ray[0]
    ry = rot_ray[1]
    rz = rot_ray[2]

    # Center position
    # For translation only:
    # - transform from ReplicaSDK coordinate system (which is RDF (right,down,forward))
    # - into our coordinate system, which is RUB (right,up,back))
    point = tf.stack([cx, cy, cz, tf.ones_like(cx)], axis=0)
    orig_shape = point.get_shape().as_list()
    point = tf.reshape(point, [4, -1])
    rot_point = tf.reshape(tf.matmul(pos, point), orig_shape)
    cx = rot_point[0]
    cy = rot_point[1]
    cz = rot_point[2]

    # Solve for ray intersection
    a = rx * rx + ry * ry + rz * rz
    b = 2 * (rx * cx + ry * cy + rz * cz)
    c = cx * cx + cy * cy + cz * cz - radius * radius
    disc = tf.square(b) - 4 * a * c

    t = (-b + tf.sqrt(disc)) / (2 * a) # This should be the only solution
    x = cx + t * rx
    y = cy + t * ry
    z = cz + t * rz

    # Project
    points = (x, y, z)
    
    return project_spherical(points, 1, None, None, width, height)

def intersect_ods(pose, center, order, intrinsics, radius, num_planes, num_batch, width, height, epsilon=1e-12):
    '''
    @Args:
    -pose: camera pose [1,4,4]
    -center: the target center of projection
    -order: 1 if left ods, -1 if right ods
    -intrinsics: the first element is the baseline
    -radius: list of msi radius
    '''

    # Pixels in the target image
    S, T = lat_long_grid((height, width))
    S = tf.tile(tf.expand_dims(S, 0), [num_planes, 1, 1])
    T = tf.tile(tf.expand_dims(T, 0), [num_planes, 1, 1])

    # Reshape radius
    radius = tf.reshape(radius, [num_planes, 1, 1])
    baseline = intrinsics[0][0][0]

    # Ray direction
    cosT = tf.cos(T)
    rx = tf.cos(S) * cosT
    ry = tf.sin(T)
    rz = - tf.sin(S) * cosT

    # Ray center
    cx = - tf.sin(S) * ( baseline ) * order
    cy = tf.zeros_like(S)
    cz = - tf.cos(S) * ( baseline ) * order

    # Transform the ray using pose
    (rx, ry, rz), (cx, cy, cz) = transform_ray((rx, ry, rz), (cx, cy, cz), pose)

    # Solve for ray intersection
    points = get_sphere_intersections((rx, ry, rz), (cx, cy, cz), radius)

    # Project into ERP images
    return project_spherical(points, order, None, intrinsics, width, height)

def intersect_perspective(pos, center, radius, num_planes, num_batch,
                          width, height, tgt_width, tgt_height, intrinsics, epsilon=1e-12):
    '''
    @args:
    -pos: target pose [4, 4]
    -center: center of projection [3, 1]
    '''
    # Pixels in the target image
    S, T = uv_grid((tgt_height, tgt_width))
    S = tf.tile(tf.expand_dims(S, 0), [num_planes, 1, 1])
    T = tf.tile(tf.expand_dims(T, 0), [num_planes, 1, 1])

    # Reshape radius
    radius = tf.reshape(radius, [num_planes, 1, 1])

    # Pixel coordinates
    # TODO: The intrinsics are hardcoded, see backproject_planar for how to use
    # intrinsics if available
    rx = S * 0.1
    ry = T * 0.05
    rz = -tf.ones_like(S) * 0.05

    # Center
    cx = center[0]
    cy = center[1]
    cz = -center[2]

    # Transform the ray using pose
    (rx, ry, rz), (cx, cy, cz) = transform_ray((rx, ry, rz), (cx, cy, cz), pos)

    # Solve for ray intersection
    points = get_sphere_intersections((rx, ry, rz), (cx, cy, cz), radius)

    # Get pixel coorindates for ERP lookup
    return project_spherical(points, 1, None, None, width, height)
