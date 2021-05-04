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

from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
import geometry.projector as pj
from nets import gcn_net, msi_train_net, msi_coord_train_net, msi_inference_net, msi_coord_inference_net
from matryodshka.utils import write_image
import elpips.elpips as elpips

flags = tf.app.flags
FLAGS = flags.FLAGS

class MSI(object):
  """
  Class definition for MPI learning module.
  """
  def __init__(self):
    pass

  def infer_msi(self,
                prev_msi,
                raw_src_image,
                raw_ref_image,
                raw_hres_src_image,
                raw_hres_ref_image,
                ref_pose,
                src_pose,
                tgt_pose,
                intrinsics,
                which_color_pred,
                num_msi_planes,
                psv_planes,
                extra_outputs='',
                ngf=64):
    """Construct the msi inference graph.
    Args:
      raw_src_image: source images [batch, height, width, 3]
      raw_ref_image: reference image [batch, height, width, 3]
      ref_pose: reference frame pose (world to camera) [batch, 4, 4]
      src_pose: source frame poses (world to camera) [batch, 4, 4]
      intrinsics: camera intrinsics [batch, 3, 3]
      which_color_pred: method for predicting the color at each msi plane (see README)
      num_msi_planes: number of msi planes to predict
      psv_planes: list of depth of plane sweep volume (PSV) planes
      extra_outputs: extra variables to output in addition to RGBA layers
    Returns:
      outputs: a collection of output tensors.
    """

    # Preprocessing
    batch_size, img_height, img_width, _ = raw_src_image.get_shape().as_list()
    if 'hrestgt' in FLAGS.supervision:
        _, hres_img_height, hres_img_width, _ = raw_hres_src_image.get_shape().as_list()

    with tf.name_scope('preprocessing'):
        src_image = self.preprocess_image(raw_src_image)
        ref_image = self.preprocess_image(raw_ref_image)
        if 'hrestgt' in FLAGS.supervision:
            hres_src_image = self.preprocess_image(raw_hres_src_image)
            hres_ref_image = self.preprocess_image(raw_hres_ref_image)

    with tf.name_scope('format_network_input'):

        reshaped_msi = tf.reshape(prev_msi, [1, 320, 640, 32*4])

        if FLAGS.input_type == 'REALESTATE_PP':
            net_input = self.format_realestate_network_input(ref_image, src_image, 
                                                             ref_pose, src_pose,
                                                             psv_planes, intrinsics)
        else:
            net_input = self.format_network_input_synth(ref_image, src_image, tgt_pose,
                                                  ref_pose, src_pose,
                                                  psv_planes, intrinsics)
            net_input = tf.concat([net_input, reshaped_msi], axis=-1)
        if 'hrestgt' in FLAGS.supervision:
            hres_net_input = self.format_network_input(hres_ref_image, hres_src_image,
                                                       ref_pose, src_pose,
                                                       psv_planes, intrinsics)

    # get the correct network for current run
    if FLAGS.operation == 'train':
        if FLAGS.coord_net:
            msi_net = msi_coord_train_net
        else:
            msi_net = msi_train_net

    elif FLAGS.operation == 'export':
        if FLAGS.coord_net:
            msi_net = msi_coord_inference_net
        else:
            msi_net = msi_inference_net

    # predict rgba layers
    '''
        @color prediction schemes:
        -blend_psv [default]: blend both psv + alpha
            - num_msi_planes x 2
        -blend_bg: blend a single psv with a background layer + alpha
            - num_msi_planes x 2 + 3
        -bleng_bg_psv: ( ( blend both psv ) + blend with a background ) + alpha
            - num_msi_planes x 3 + 3
        -alpha_only: use psv rgb + alpha for rgba layers composition
            - num_msi_planes
    '''
    if which_color_pred == 'blend_psv':
        if FLAGS.transform_inverse_reg:
          if not FLAGS.jitter:
              msi_pred = msi_net(net_input, num_msi_planes * 2, ngf=FLAGS.ngf, reuse_weights=False)
          else:
              msi_pred = msi_net(net_input, num_msi_planes * 2, ngf=FLAGS.ngf, reuse_weights=True)
        else:
            # adding (num_msi_planes) to num_outputs for predicted blending weights
            msi_pred = msi_net(net_input, num_msi_planes * 3, ngf=FLAGS.ngf, reuse_weights=tf.AUTO_REUSE)
        if FLAGS.net_only:
          #for model export
          return None
        with tf.name_scope('layer_prediction'):
            # Rescale blend_weights/alpha to (0, 1)
            blend_weights = (msi_pred[:, :, :, :num_msi_planes] + 1.) / 2.
            alphas = (msi_pred[:, :, :, num_msi_planes:num_msi_planes * 2] + 1.) / 2.
            msi_blend_weights = (msi_pred[:, :, :, num_msi_planes * 2:num_msi_planes * 3] + 1.) / 2.

            # Assemble into an msi (rgba_layers)
            for i in range(num_msi_planes):
                # rgb from inputs (foreground and background)
                fg_rgb = net_input[:, :, :, i * 3:(1 + i) * 3]
                bg_rgb = net_input[:, :, :, (num_msi_planes + i) * 3:(num_msi_planes + 1 + i) * 3]
                
                # rgb from current msi
                msi_rgb = prev_msi[:, :, :, i, :3]
                msi_a = prev_msi[:, :, :, i, 3:]

                # blend psv inputs
                w = tf.expand_dims(blend_weights[:, :, :, i], -1)
                rgb_psv = w * fg_rgb + (1 - w) * bg_rgb

                # blend with current msi
                w_msi = tf.expand_dims(msi_blend_weights[:, :, :, i], -1)
                curr_rgb = w_msi * msi_rgb + (1 - w_msi) * rgb_psv

                # update alpha
                curr_alpha = tf.expand_dims(alphas[:, :, :, i], -1)
                curr_alpha = curr_alpha * (1/(FLAGS.num_synth+1)) + (1 - 1/(FLAGS.num_synth+1))*msi_a

                curr_rgba = tf.concat([curr_rgb, curr_alpha], axis=3)
                if i == 0:
                    rgba_layers = curr_rgba
                else:
                    rgba_layers = tf.concat([rgba_layers, curr_rgba], axis=3)
            rgba_layers = tf.reshape(rgba_layers, [batch_size, img_height, img_width, num_msi_planes, 4])

            #Upsampled rgba layers
            if 'hrestgt' in FLAGS.supervision:
                upsampled_blend_weights = tf.image.resize(blend_weights, [FLAGS.hres_height, FLAGS.hres_width], align_corners=True, method=tf.image.ResizeMethod.BILINEAR)
                upsampled_alphas = tf.image.resize(alphas, [FLAGS.hres_height, FLAGS.hres_width], align_corners=True, method=tf.image.ResizeMethod.BILINEAR)
                # Assemble into an msi (rgba_layers)
                for i in range(num_msi_planes):
                    ufg_rgb = hres_net_input[:, :, :, i * 3:(1 + i) * 3]
                    ubg_rgb = hres_net_input[:, :, :, (32 + i) * 3:(33 + i) * 3]
                    ucurr_alpha = tf.expand_dims(upsampled_alphas[:, :, :, i], -1)
                    uw = tf.expand_dims(upsampled_blend_weights[:, :, :, i], -1)
                    ucurr_rgb = uw * ufg_rgb + (1 - uw) * ubg_rgb
                    ucurr_rgba = tf.concat([ucurr_rgb, ucurr_alpha], axis=3)
                    if i == 0:
                        urgba_layers = ucurr_rgba
                    else:
                        urgba_layers = tf.concat([urgba_layers, ucurr_rgba], axis=3)
                urgba_layers = tf.reshape(urgba_layers, [batch_size, hres_img_height, hres_img_width, num_msi_planes, 4])
    elif which_color_pred == 'blend_bg':
        if FLAGS.transform_inverse_reg:
            if not FLAGS.jitter:
                msi_pred = msi_net(net_input, 3 + num_msi_planes * 2, ngf=FLAGS.ngf, reuse_weights=False)
            else:
                msi_pred = msi_net(net_input, 3 + num_msi_planes * 2, ngf=FLAGS.ngf, reuse_weights=True)
        else:
            msi_pred = msi_net(net_input, 3 + num_msi_planes * 2, ngf=FLAGS.ngf, reuse_weights=tf.AUTO_REUSE)
        if FLAGS.net_only:
            return None
        with tf.name_scope('layer_prediction'):
            # Rescale blend_weights to (0, 1)
            blend_weights = (msi_pred[:, :, :, :num_msi_planes] + 1.) / 2.
            # Rescale alphas to (0, 1)
            alphas = (msi_pred[:, :, :, num_msi_planes:num_msi_planes * 2] + 1.) / 2.
            bg_rgb = msi_pred[:, :, :, -3:]

            # Assemble into an msi (rgba_layers)
            for i in range(num_msi_planes):
                fg_rgb = net_input[:, :, :, i * 3:(1 + i) * 3]
                curr_alpha = tf.expand_dims(alphas[:, :, :, i], -1)
                w = tf.expand_dims(blend_weights[:, :, :, i], -1)
                curr_rgb = w * fg_rgb + (1 - w) * bg_rgb
                curr_rgba = tf.concat([curr_rgb, curr_alpha], axis=3)
                if i == 0:
                    rgba_layers = curr_rgba
                else:
                    rgba_layers = tf.concat([rgba_layers, curr_rgba], axis=3)
            rgba_layers = tf.reshape(rgba_layers, [batch_size, img_height, img_width, num_msi_planes, 4])

            #Upsampled rgba layers
            if 'hrestgt' in FLAGS.supervision:
                upsampled_blend_weights = tf.image.resize(blend_weights, [FLAGS.hres_height, FLAGS.hres_width], align_corners=True, method=tf.image.ResizeMethod.BILINEAR)
                upsampled_alphas = tf.image.resize(alphas, [FLAGS.hres_height, FLAGS.hres_width], align_corners=True, method=tf.image.ResizeMethod.BILINEAR)
                ubg_rgb = tf.image.resize(bg_rgb, [FLAGS.hres_height, FLAGS.hres_width], align_corners=True, method=tf.image.ResizeMethod.BILINEAR)
                # Assemble into an msi (rgba_layers)
                for i in range(num_msi_planes):
                    ufg_rgb = hres_net_input[:, :, :, i * 3:(1 + i) * 3]
                    ucurr_alpha = tf.expand_dims(upsampled_alphas[:, :, :, i], -1)
                    uw = tf.expand_dims(upsampled_blend_weights[:, :, :, i], -1)
                    ucurr_rgb = uw * ufg_rgb + (1 - uw) * ubg_rgb
                    ucurr_rgba = tf.concat([ucurr_rgb, ucurr_alpha], axis=3)
                    if i == 0:
                        urgba_layers = ucurr_rgba
                    else:
                        urgba_layers = tf.concat([urgba_layers, ucurr_rgba], axis=3)
                urgba_layers = tf.reshape(urgba_layers, [batch_size, hres_img_height, hres_img_width, num_msi_planes, 4])
    elif which_color_pred == 'blend_bg_psv':
        if FLAGS.transform_inverse_reg:
            if not FLAGS.jitter:
                msi_pred = msi_net(net_input, 3 + num_msi_planes * 3, ngf=FLAGS.ngf, reuse_weights=False)
            else:
                msi_pred = msi_net(net_input, 3 + num_msi_planes * 3, ngf=FLAGS.ngf, reuse_weights=True)
        else:
            msi_pred = msi_net(net_input, 3 + num_msi_planes * 3, ngf=FLAGS.ngf, reuse_weights=tf.AUTO_REUSE)
        if FLAGS.net_only:
            return None
        with tf.name_scope('layer_prediction'):
            # Rescale blend_weights to (0, 1)
            blend_weights = (msi_pred[:, :, :, :num_msi_planes] + 1.) / 2.
            bg_blend_weights = (msi_pred[:, :, :, num_msi_planes * 2 :num_msi_planes * 3] + 1.) / 2.
            # Rescale alphas to (0, 1)
            alphas = (msi_pred[:, :, :, num_msi_planes:num_msi_planes * 2] + 1.) / 2.
            pred_bg = msi_pred[:, :, :, -3:]

            # Assemble into an msi (rgba_layers)
            for i in range(num_msi_planes):
              #blend two psv first
              fg_rgb = net_input[:, :, :, i * 3:(1 + i) * 3]
              bg_rgb = net_input[:, :, :, (num_msi_planes+i) * 3:(num_msi_planes+1+i) * 3]
              w = tf.expand_dims(blend_weights[:, :, :, i], -1)
              curr_rgb = w * fg_rgb + (1 - w) * bg_rgb
              #blend with predicted bg layer
              bg_w = tf.expand_dims(bg_blend_weights[:, :, :, i], -1)
              curr_rgb = bg_w * curr_rgb + (1 - bg_w) * pred_bg
              curr_alpha = tf.expand_dims(alphas[:, :, :, i], -1)
              curr_rgba = tf.concat([curr_rgb, curr_alpha], axis=3)
              if i == 0:
                rgba_layers = curr_rgba
              else:
                rgba_layers = tf.concat([rgba_layers, curr_rgba], axis=3)
            rgba_layers = tf.reshape(rgba_layers, [batch_size, img_height, img_width, num_msi_planes, 4])

            #TODO: Upsampled rgba layers
    elif which_color_pred == 'alpha_only':
        if FLAGS.transform_inverse_reg:
            if not FLAGS.jitter:
                msi_pred = msi_net(net_input, num_msi_planes, ngf=FLAGS.ngf, reuse_weights=False)
            else:
                msi_pred = msi_net(net_input, num_msi_planes, ngf=FLAGS.ngf, reuse_weights=True)
        else:
            msi_pred = msi_net(net_input, num_msi_planes, ngf=FLAGS.ngf, reuse_weights=tf.AUTO_REUSE)
        if FLAGS.net_only:
            return None
        with tf.name_scope('layer_prediction'):
            # Rescale alphas to (0, 1)
            alphas = (msi_pred[:, :, :, :num_msi_planes] + 1.) / 2.
            # Assemble into an msi (rgba_layers)
            for i in range(num_msi_planes):
              curr_rgb = net_input[:, :, :, i * 3:(1 + i) * 3]
              curr_alpha = tf.expand_dims(alphas[:, :, :, i], -1)
              curr_rgba = tf.concat([curr_rgb, curr_alpha], axis=3)
              if i == 0:
                rgba_layers = curr_rgba
              else:
                rgba_layers = tf.concat([rgba_layers, curr_rgba], axis=3)
            rgba_layers = tf.reshape(rgba_layers, [batch_size, img_height, img_width, num_msi_planes, 4])
            #TODO: Upsampled rgba layers

    # collect outputs to return
    pred = {}
    pred['rgba_layers'] = rgba_layers
    pred['msi_blend_weights'] = msi_blend_weights
    if 'hrestgt' in FLAGS.supervision:
        pred['hres_rgba_layers'] = urgba_layers
    if 'blend_weights' in extra_outputs:
        if 'blend' in FLAGS.which_color_pred:
            pred['blend_weights'] = blend_weights
            if 'bg' in FLAGS.which_color_pred:
                pred['bg_blend_weights'] = bg_blend_weights
    if 'alpha' in extra_outputs:
      pred['alphas'] = alphas
    if 'psv' in extra_outputs:
      pred['psv'] = net_input[:, :, :, :]
    return pred, net_input

  def infer_gcn_msi(self,
                    raw_src_image,
                    raw_ref_image,
                    ref_pose,
                    src_pose,
                    intrinsics,
                    which_color_pred,
                    num_msi_planes,
                    psv_planes,
                    inputs,
                    extra_outputs='',
                    ngf=64):
      """Construct the msi inference graph for gcn.
      Args:
        raw_src_image: source images [batch, height, width, 3]
        raw_ref_image: reference image [batch, height, width, 3]
        ref_pose: reference frame pose (world to camera) [batch, 4, 4]
        src_pose: source frame poses (world to camera) [batch, 4, 4]
        intrinsics: camera intrinsics [batch, 3, 3]
        which_color_pred: method for predicting the color at each msi plane (see README)
        num_msi_planes: number of msi planes to predict
        psv_planes: list of depth of plane sweep volume (PSV) planes
        extra_outputs: extra variables to output in addition to RGBA layers
      Returns:
        outputs: a collection of output tensors.
      """

      # Preprocessing
      support = inputs['support']
      p2v = inputs['p2v']

      batch_size, img_height, img_width, _ = raw_src_image.get_shape().as_list()

      with tf.name_scope('preprocessing'):
          src_image = self.preprocess_image(raw_src_image)
          ref_image = self.preprocess_image(raw_ref_image)

      with tf.name_scope('format_network_input'):
          net_input = self.format_gcn_network_input(ref_image, src_image,
                                                    ref_pose, src_pose,
                                                    psv_planes, intrinsics)
          net_input_images = self.format_network_input(ref_image, src_image,
                                                       ref_pose, src_pose,
                                                       psv_planes, intrinsics)
      # get the correct network for current run
      msi_net = gcn_net
      # predict rgba layers
      '''
          @color prediction schemes:
          -blend_psv [default]: blend both psv + alpha
              - num_msi_planes x 2
          -blend_bg: blend a single psv with a background layer + alpha
              - num_msi_planes x 2 + 3
          -bleng_bg_psv: ( ( blend both psv ) + blend with a background ) + alpha
              - num_msi_planes x 3 + 3
          -alpha_only: use psv rgb + alpha for rgba layers composition
              - num_msi_planes
      '''
      if which_color_pred == 'blend_psv':
          msi_pred = msi_net(net_input, num_msi_planes * 2, support, ngf=FLAGS.ngf)
          msi_pred = pj.mesh_to_equirect(msi_pred, p2v)

          with tf.name_scope('layer_prediction'):
              # Rescale blend_weights/alpha to (0, 1)
              blend_weights = (msi_pred[:, :, :, :num_msi_planes] + 1.) / 2.
              alphas = (msi_pred[:, :, :, num_msi_planes:num_msi_planes * 2] + 1.) / 2.

              # Assemble into an msi (rgba_layers)
              for i in range(num_msi_planes):
                  fg_rgb = net_input_images[:, :, :, i * 3:(1 + i) * 3]
                  bg_rgb = net_input_images[:, :, :, (num_msi_planes + i) * 3:(num_msi_planes + 1 + i) * 3]
                  curr_alpha = tf.expand_dims(alphas[:, :, :, i], -1)
                  w = tf.expand_dims(blend_weights[:, :, :, i], -1)
                  curr_rgb = w * fg_rgb + (1 - w) * bg_rgb
                  curr_rgba = tf.concat([curr_rgb, curr_alpha], axis=3)
                  if i == 0:
                      rgba_layers = curr_rgba
                  else:
                      rgba_layers = tf.concat([rgba_layers, curr_rgba], axis=3)
              rgba_layers = tf.reshape(rgba_layers, [batch_size, img_height, img_width, num_msi_planes, 4])

      # collect outputs to return
      pred = {}
      pred['rgba_layers'] = rgba_layers
      if 'blend_weights' in extra_outputs:
          if 'blend' in FLAGS.which_color_pred:
              pred['blend_weights'] = blend_weights
      if 'alpha' in extra_outputs:
          pred['alphas'] = alphas
      if 'psv' in extra_outputs:
          pred['psv'] = net_input_images[:, :, :, :]
      return pred, net_input_images

  def msi_render_equirect_depth(self, rgba_layers, tgt_pose_rt, tgt_pos, planes, intrinsics):
    """
    Render a target view from an msi representation.
    Args:
      rgba_layers: input msi [batch, height, width, #planes, 4]
      planes: list of depth for each plane
    Returns:
      rendered view [batch, height, width, 3]
    """

    batch_size, _, _ = tgt_pose_rt.get_shape().as_list()
    depths = tf.constant(planes, shape=[len(planes), 1])
    depths = tf.tile(depths, [1, batch_size])
    rgba_layers = tf.transpose(rgba_layers, [3, 0, 1, 2, 4])

    proj_images = pj.projective_forward_sphere(rgba_layers, intrinsics, tgt_pose_rt, tgt_pos, depths)

    proj_images_list = []
    for i in range(len(planes)):
      proj_images_list.append(proj_images[i])
    output_image = pj.over_composite_depth(proj_images_list)
    return output_image

  def msi_render_equirect_view(self, rgba_layers, tgt_pose_rt, tgt_pos, planes, intrinsics):
    """
    Render a target view from an MSI representation.
    Args:
      rgba_layers: input msi [batch, height, width, #planes, 4]
      tgt_pose_rt: target pose in the form of [R,t], [batch, 4, 4]
      tgt_pos: the input target offset [batch, 3, 1]
      planes: list of depth for each plane
    Returns:
      rendered view [batch, height, width, 3]
    """

    batch_size, _, _ = tgt_pose_rt.get_shape().as_list()
    depths = tf.constant(planes, shape=[len(planes), 1])
    depths = tf.tile(depths, [1, batch_size])
    rgba_layers = tf.transpose(rgba_layers, [3, 0, 1, 2, 4])
    proj_images = pj.projective_forward_sphere(rgba_layers, intrinsics, tgt_pose_rt, tgt_pos, depths)
    proj_images_list = []
    for i in range(len(planes)):
      proj_images_list.append(proj_images[i])
    output_image = pj.over_composite(proj_images_list)

    return output_image

  def msi_render_equirect_view_single(self, rgba_layers, tgt_pose_rt, tgt_pos, planes, intrinsics):
    """
    Render a target view from an single MSI layer for faster rerendering.
    Args:
      rgba_layers: input msi [batch, height, width, #planes, 4]
      tgt_pose_rt: target pose in the form of [R,t], [batch, 4, 4]
      tgt_pos: the input target offset [batch, 3, 1]
      planes: list of depth for each plane
    Returns:
      rendered view [batch, height, width, 3]
    """

    batch_size, _, _ = tgt_pose_rt.get_shape().as_list()
    if isinstance(planes, list):
        depths = tf.constant(planes, shape=[len(planes), 1])
    else:
        num_planes, = planes.get_shape().as_list()
        depths = tf.reshape(planes, [num_planes,1])
    depths = tf.tile(depths, [1, batch_size])
    rgba_layers = tf.transpose(rgba_layers, [3, 0, 1, 2, 4])
    proj_images = pj.projective_forward_sphere(rgba_layers, intrinsics, tgt_pose_rt, tgt_pos, depths)
    return proj_images

  def msi_render_equirect_depth_single(self, rgba_layers, tgt_pose_rt, tgt_pos, planes, intrinsics):
    """
    Render a target view from an msi representation.
    Args:
      rgba_layers: input msi [batch, height, width, #planes, 4]
      planes: list of depth for each plane
    Returns:
      rendered view [batch, height, width, 3]
    """

    batch_size, _, _ = tgt_pose_rt.get_shape().as_list()
    if isinstance(planes, list):
        depths = tf.constant(planes, shape=[len(planes), 1])
    else:
        num_planes, = planes.get_shape().as_list()
        depths = tf.reshape(planes, [num_planes,1])
    depths = tf.tile(depths, [1, batch_size])
    rgba_layers = tf.transpose(rgba_layers, [3, 0, 1, 2, 4])
    proj_images = pj.projective_forward_sphere(rgba_layers, intrinsics, tgt_pose_rt, tgt_pos, depths)
    return proj_images

  def msi_render_perspective_view(self, rgba_layers, tgt_pose_rt, tgt_pos, planes, intrinsics,
                                  viewing_window=3, psp_height=270, psp_width=480):
    """
    Render a target view from an msi representation.
    Args:
      rgba_layers: input msi [batch, height, width, #planes, 4]
      tgt_pose_rt: target pose in the form of [R,t], [batch, 4, 4]
      tgt_pos: the input target offset [batch, 3, 1]
      planes: list of depth for each plane
    Returns:
      rendered view [batch, height, width, 3]
    """

    batch_size, _, _ = tgt_pose_rt.get_shape().as_list()
    depths = tf.constant(planes, shape=[len(planes), 1])
    depths = tf.tile(depths, [1, batch_size])
    rgba_layers = tf.transpose(rgba_layers, [3, 0, 1, 2, 4])

    proj_images = pj.projective_forward_sphere_to_perspective(rgba_layers, intrinsics, tgt_pose_rt, tgt_pos, depths, viewing_window, psp_height, psp_width)

    proj_images_list = []
    for i in range(len(planes)):
      proj_images_list.append(proj_images[i])
    output_image = pj.over_composite(proj_images_list)

    return output_image

  def msi_render_ods_view(self, rgba_layers, order, jitter_pose, tgt_pos, planes, intrinsics):
    """Render a target view from an msi representation.

    Args:
      rgba_layers: input msi [batch, height, width, #planes, 4]
      jitter_pose: camera pose [1,4,4]
      tgt_pos: target position to render from [batch, 3]
      planes: list of depth for each plane
    Returns:
      rendered view [batch, height, width, 3]
    """

    batch_size, _, _, = intrinsics.shape
    depths = tf.constant(planes, shape=[len(planes), 1])
    depths = tf.tile(depths, [1, batch_size])
    rgba_layers = tf.transpose(rgba_layers, [3, 0, 1, 2, 4])
    proj_images = pj.projective_forward_ods(rgba_layers, order, intrinsics, jitter_pose, tgt_pos, depths)

    proj_images_list = []
    for i in range(len(planes)):
      proj_images_list.append(proj_images[i])
    output_image = pj.over_composite(proj_images_list)

    return output_image

  def mpi_render_view(self, rgba_layers, tgt_pose, planes, intrinsics):
    """Render a target perspective view from an msi representation. ( From original matryodshka repo. )

    Args:
      rgba_layers: input msi [batch, height, width, #planes, 4]
      tgt_pose: target pose to render from [batch, 4, 4]
      planes: list of depth for each plane
      intrinsics: camera intrinsics [batch, 3, 3]
    Returns:
      rendered view [batch, height, width, 3]
    """
    batch_size, _, _ = tgt_pose.get_shape().as_list()
    depths = tf.constant(planes, shape=[len(planes), 1])
    depths = tf.tile(depths, [1, batch_size])
    rgba_layers = tf.transpose(rgba_layers, [3, 0, 1, 2, 4])
    proj_images = pj.projective_forward_homography(rgba_layers, intrinsics,
                                                   tgt_pose, depths)
    proj_images_list = []
    for i in range(len(planes)):
      proj_images_list.append(proj_images[i])
    output_image = pj.over_composite(proj_images_list)
    return output_image

  def build_train_graph(self,
                        inputs,
                        min_depth,
                        max_depth,
                        num_psv_planes,
                        num_msi_planes,
                        which_color_pred='bg',
                        which_loss='pixel',
                        learning_rate=0.0002,
                        beta1=0.9):

    """Construct the training computation graph.

    Args:
      inputs: dictionary of tensors (see 'input_data' below) needed for training
      min_depth: minimum depth for the plane sweep volume (PSV) and msi/MSI planes
      max_depth: maximum depth for the PSV and msi/MSI planes
      num_psv_planes: number of PSV planes for network input
      num_msi_planes: number of msi/MSI planes to infer
      which_color_pred: how to predict the color at each msi/MSI plane
      which_loss: which loss function to use (pixel or elpips)
      learning_rate: learning rate
      beta1: hyperparameter for ADAM
    Returns:
      A train_op to be used for training.
    """
    with tf.name_scope('setup'):
      psv_planes = self.inv_depths(min_depth, max_depth, num_psv_planes)
      msi_planes = self.inv_depths(min_depth, max_depth, num_msi_planes)
      prev_msi = tf.placeholder(tf.float32, shape=(1, 320, 640, num_msi_planes, 4), name='prev_msi')
      view = tf.placeholder(tf.int32, name='view')
      orig_pose = tf.placeholder(tf.float32, shape=(1, 4, 4), name='orig_pose')
      first_tgt = tf.placeholder(tf.float32, shape=(1, 320, 640, 3), name='first_tgt')
      first_tgt_pose = tf.placeholder(tf.float32, shape=(1, 3), name='first_tgt_pose')
    with tf.name_scope('input_data'):
      raw_tgt_image = tf.cond(tf.math.equal(view, tf.constant(0)), lambda:inputs['tgt_image'], lambda:first_tgt)
      raw_ref_image = inputs['ref_image']
      raw_src_image = inputs['src_image']
      raw_hres_tgt_image = inputs['hres_tgt_image'] if 'hrestgt' in FLAGS.supervision else None
      raw_hres_ref_image = inputs['hres_ref_image'] if 'hrestgt' in FLAGS.supervision else None
      raw_hres_src_image = inputs['hres_src_image'] if 'hrestgt' in FLAGS.supervision else None

      tgt_pose = inputs['tgt_pose']
      first_tgt_pose = tf.cond(tf.math.equal(view, tf.constant(0)), lambda:inputs['tgt_pose'], lambda:first_tgt_pose)
      ref_pose = inputs['ref_pose']
      orig_pose = tf.cond(tf.math.equal(view, tf.constant(0)), lambda:ref_pose, lambda:orig_pose)
      src_pose = inputs['src_pose']
      intrinsics = inputs['intrinsics']
      if FLAGS.gcn:
          support = inputs['support']
          coord = inputs['coord']
          p2v = inputs['p2v']

    with tf.name_scope('inference'):
      assert FLAGS.jitter == False
      if FLAGS.gcn:
          pred, net_input = self.infer_gcn_msi(raw_src_image, raw_ref_image, ref_pose, src_pose,
                                               intrinsics, which_color_pred, FLAGS.num_msi_planes, psv_planes, inputs)
      else:
          pred, net_input = self.infer_msi(prev_msi, raw_src_image, raw_ref_image, raw_hres_src_image, raw_hres_ref_image, ref_pose, src_pose, orig_pose,
                                            intrinsics, which_color_pred, FLAGS.num_msi_planes, psv_planes, inputs)
      rgba_layers = pred['rgba_layers']
      msi_blend_weights = pred['msi_blend_weights']

      if 'hrestgt' in FLAGS.supervision:
          hrgba_layers = pred['hres_rgba_layers']

      #gather relevant tensor into the inputs dictionary
      inputs['unjitterd_net_input']  = net_input

      if FLAGS.transform_inverse_reg:
        FLAGS.jitter = True
        pred_jitter, net_input_jitter = self.infer_msi(raw_src_image, raw_ref_image, raw_hres_src_image, raw_hres_ref_image, ref_pose, src_pose,
                                                       intrinsics, which_color_pred, num_msi_planes, psv_planes, inputs)
        rgba_layers_jitter = pred_jitter['rgba_layers']
        #can't fit highres transform inverse training into memory so no high-res part
        inputs['jittered_net_input'] = net_input_jitter

    with tf.name_scope('synthesis'):
        if FLAGS.input_type == 'ODS':
            if 'tgt' in FLAGS.supervision:
                output_image = self.msi_render_equirect_view(rgba_layers, tf.expand_dims(tf.eye(4), axis=0), first_tgt_pose, msi_planes, intrinsics)
                output_depth_image = self.msi_render_equirect_depth(rgba_layers, tf.expand_dims(tf.eye(4), axis=0), first_tgt_pose, msi_planes, intrinsics)
            if 'hrestgt' in FLAGS.supervision:
                hres_output_image = self.msi_render_equirect_view(hrgba_layers, tf.expand_dims(tf.eye(4), axis=0), first_tgt_pose, msi_planes, intrinsics)
                hres_output_depth_image = self.msi_render_equirect_depth(hrgba_layers, tf.expand_dims(tf.eye(4), axis=0), first_tgt_pose, msi_planes, intrinsics)
            if 'src' in FLAGS.supervision:
                src_output_image = self.msi_render_ods_view(rgba_layers, -1, tf.expand_dims(tf.eye(4), axis=0), first_tgt_pose, msi_planes, intrinsics)
            if 'ref' in FLAGS.supervision:
                ref_output_image = self.msi_render_ods_view(rgba_layers, 1, tf.expand_dims(tf.eye(4), axis=0), first_tgt_pose, msi_planes, intrinsics)

            if FLAGS.transform_inverse_reg:
                jitter_pose = tf.get_default_graph().get_tensor_by_name("jitter_pose:0")
                if 'tgt' in FLAGS.supervision:
                    jitter_output_image = self.msi_render_equirect_view(rgba_layers_jitter, jitter_pose, tgt_pose, msi_planes, intrinsics)
                    jitter_output_depth_image = self.msi_render_equirect_depth(rgba_layers_jitter, jitter_pose, tgt_pose, msi_planes, intrinsics)
                if 'src' in FLAGS.supervision:
                    jsrc_output_image = self.msi_render_ods_view(rgba_layers, -1, jitter_pose, tgt_pose, msi_planes, intrinsics)
                if 'ref' in FLAGS.supervision:
                    jref_output_image = self.msi_render_ods_view(rgba_layers, 1, jitter_pose, tgt_pose, msi_planes, intrinsics)
                #hres jitter output not supported as the model gets too large.
        else:
            ref_pose_inv = tf.get_default_graph().get_tensor_by_name("interp_pose_inv:0")
            rel_pose = tf.matmul(tgt_pose, ref_pose_inv)
            output_image = self.mpi_render_view(rgba_layers, rel_pose, msi_planes, intrinsics)
            if FLAGS.transform_inverse_reg:
                jitter_pose_inv = tf.get_default_graph().get_tensor_by_name("jitter_pose_inv:0")
                ref_pose_inv = tf.matmul(ref_pose_inv, jitter_pose_inv)
                rel_pose = tf.matmul(tgt_pose, ref_pose_inv)
                jitter_output_image = self.mpi_render_view(rgba_layers_jitter, rel_pose, msi_planes, intrinsics)
    with tf.name_scope('loss'):

        #loss functions
        metric = elpips.Metric(elpips.elpips_vgg(batch_size=1),back_prop=True)
        def get_loss(y,p,loss_type,spherical_attention):
            if spherical_attention:
                sph_weights = self.create_spherical_weights()
                y = tf.multiply(sph_weights, y)
                p = tf.multiply(sph_weights, p)
            if loss_type == 'pixel':
                return tf.reduce_mean(tf.nn.l2_loss(p-y))
            elif loss_type == 'elpips':
                return tf.reduce_mean(metric.forward(p, y))

        #get raw ground-truth input images
        tgt_image = self.preprocess_image(raw_tgt_image)
        src_image = self.preprocess_image(raw_src_image)
        ref_image = self.preprocess_image(raw_ref_image)
        if 'hrestgt' in FLAGS.supervision:
            hres_tgt_image = self.preprocess_image(raw_hres_tgt_image)

        total_loss = 0.
        if which_loss == 'pixel':
            if FLAGS.transform_inverse_reg:
                if 'tgt' in FLAGS.supervision:
                    reg_reconstruction_loss = get_loss(output_image, tgt_image, 'pixel', FLAGS.spherical_attention) #tf.reduce_mean(tf.nn.l2_loss(output_image - tgt_image))
                    jitter_reconstruction_loss = get_loss(jitter_output_image, tgt_image, 'pixel', FLAGS.spherical_attention) #tf.reduce_mean(tf.nn.l2_loss(jitter_output_image - tgt_image))
                    enforcement_loss = get_loss(jitter_output_image, output_image, 'pixel', FLAGS.spherical_attention) #tf.reduce_mean(tf.nn.l2_loss(output_image - jitter_output_image))
                    total_loss += reg_reconstruction_loss
                    total_loss += 10 * enforcement_loss
                if 'src' in FLAGS.supervision:
                    total_loss += get_loss(src_output_image, src_image, 'pixel', FLAGS.spherical_attention) #tf.reduce_mean(tf.nn.l2_loss(src_output_image - src_image))
                    total_loss += get_loss(jsrc_output_image, src_image, 'pixel', FLAGS.spherical_attention) #tf.reduce_mean(tf.nn.l2_loss(jsrc_output_image - src_image))
                if 'ref' in FLAGS.supervision:
                    total_loss += get_loss(ref_output_image, ref_image, 'pixel', FLAGS.spherical_attention) #tf.reduce_mean(tf.nn.l2_loss(ref_output_image - ref_image))
                    total_loss += get_loss(jref_output_image, ref_image, 'pixel', FLAGS.spherical_attention) #tf.reduce_mean(tf.nn.l2_loss(jref_output_image - ref_image))
            else:
                if 'tgt' in FLAGS.supervision:
                    total_loss += get_loss(output_image, tgt_image, 'pixel', FLAGS.spherical_attention)  #tf.reduce_mean(tf.nn.l2_loss(output_image - tgt_image))
                if 'hrestgt' in FLAGS.supervision:
                    total_loss += get_loss(hres_output_image, hres_tgt_image, 'pixel', FLAGS.spherical_attention)  #tf.reduce_mean(tf.nn.l2_loss(hres_output_image - hres_tgt_image))
                if 'src' in FLAGS.supervision:
                    total_loss += 0.0001 * get_loss(src_output_image, src_image, 'pixel', FLAGS.spherical_attention)  #tf.reduce_mean(tf.nn.l2_loss(src_output_image - src_image))
                if 'ref' in FLAGS.supervision:
                    total_loss += 0.0001 * get_loss(ref_output_image, ref_image, 'pixel', FLAGS.spherical_attention)  #tf.reduce_mean(tf.nn.l2_loss(ref_output_image - ref_image))
        if which_loss == 'elpips':
            if FLAGS.transform_inverse_reg:
                if 'tgt' in FLAGS.supervision:
                    reg_reconstruction_loss = get_loss(output_image, tgt_image, 'elpips', FLAGS.spherical_attention)
                    jitter_reconstruction_loss = get_loss(jitter_output_image, tgt_image, 'elpips', FLAGS.spherical_attention)
                    enforcement_loss = get_loss(jitter_output_image, output_image, 'elpips', FLAGS.spherical_attention)
                    total_loss += reg_reconstruction_loss
                    total_loss += 10 * enforcement_loss
                if 'src' in FLAGS.supervision:
                    total_loss += get_loss(src_output_image, src_image, 'elpips', FLAGS.spherical_attention)
                    total_loss += get_loss(jsrc_output_image, src_image, 'elpips', FLAGS.spherical_attention)
                if 'ref' in FLAGS.supervision:
                    total_loss += get_loss(ref_output_image, ref_image, 'elpips', FLAGS.spherical_attention)
                    total_loss += get_loss(jref_output_image, ref_image, 'elpips', FLAGS.spherical_attention)
            else:
                if 'tgt' in FLAGS.supervision:
                    total_loss += get_loss(output_image, tgt_image, 'elpips', FLAGS.spherical_attention)
                if 'hrestgt' in FLAGS.supervision:
                    total_loss += get_loss(hres_output_image, hres_tgt_image, 'elpips', FLAGS.spherical_attention)
                if 'src' in FLAGS.supervision:
                    total_loss += 0.0001 * get_loss(src_output_image, src_image, 'elpips', FLAGS.spherical_attention)
                if 'ref' in FLAGS.supervision:
                    total_loss += 0.0001 * get_loss(ref_output_image, ref_image, 'elpips', FLAGS.spherical_attention)

        if FLAGS.wreg:
            # add weight regulariazation
            vars = tf.trainable_variables()
            loss_reg = tf.add_n([tf.nn.l2_loss(v) for v in vars ]) * 0.001
            total_loss += loss_reg

    with tf.name_scope('train_op'):
      train_vars = [var for var in tf.trainable_variables()]
      optim = tf.train.AdamOptimizer(learning_rate, beta1)
      if FLAGS.mixed_precision:
          optim = tf.train.experimental.enable_mixed_precision_graph_rewrite(optim)
      grads_and_vars = optim.compute_gradients(total_loss, var_list=train_vars)
      train_op = optim.apply_gradients(grads_and_vars)

    # Tensorboard Summaries
    # Losses
    tf.summary.scalar('total_loss', total_loss)
    if FLAGS.transform_inverse_reg and 'tgt' in FLAGS.supervision:
        tf.summary.scalar('reg_reconstr_loss',reg_reconstruction_loss)
        tf.summary.scalar('jitter_reconstr_loss',jitter_reconstruction_loss)
        tf.summary.scalar('enforcement_loss',enforcement_loss)
    if FLAGS.wreg :
        tf.summary.scalar('reg_loss', loss_reg)

    # Source images
    tf.summary.image('src_image', raw_src_image)
    # Target image
    tf.summary.image('tgt_image', raw_tgt_image)
    # Reference image
    tf.summary.image('ref_image', raw_ref_image)

    # Output image
    if 'tgt' in FLAGS.supervision:
        tf.summary.image('output_image', self.deprocess_image(output_image))
        if FLAGS.transform_inverse_reg:
            tf.summary.image('jitter_output_image', self.deprocess_image(jitter_output_image))
    if 'src' in FLAGS.supervision:
        tf.summary.image('src_output_image', self.deprocess_image(src_output_image))
    if 'ref' in FLAGS.supervision:
        tf.summary.image('ref_output_image', self.deprocess_image(ref_output_image))

    # Predicted color and alpha layers
    for i in [0, 8, 16, 24, 31]:
      rgb = rgba_layers[:, :, :, i, :3]
      alpha = rgba_layers[:, :, :, i, 3:]
      msi_weights = msi_blend_weights[:,:,:,i]
      msi_weights = tf.expand_dims(msi_weights, axis=-1)
      if FLAGS.transform_inverse_reg:
          jrgb = rgba_layers_jitter[:, :, :, i, :3]
          jalpha = rgba_layers_jitter[:, :, :, i, 3:]
          tf.summary.image('jitter_rgb_layer_%d' % i, self.deprocess_image(jrgb))
          tf.summary.image('jitter_alpha_layer_%d' % i, jalpha)
          tf.summary.image('jitter_rgba_layer_%d' % i, self.deprocess_image(jrgb * jalpha))
      tf.summary.image('rgb_layer_%d' % i, self.deprocess_image(rgb))
      tf.summary.image('alpha_layer_%d' % i, alpha)
      tf.summary.image('msi_weights_%d' % i, msi_weights)
      tf.summary.image('rgba_layer_%d' % i, self.deprocess_image(rgb * alpha))

    #dry run operations for sanity check
    if FLAGS.dry_run or FLAGS.dry_run_inference:
        #make directory
        dryrun_dir = 'dryrun/' + FLAGS.experiment_name
        if not os.path.isdir(dryrun_dir):
            os.mkdir(dryrun_dir)
        #deprocess psv-formatted inputs
        net_input = tf.concat(net_input, axis=3)
        inputs['formattedInput'] = self.deprocess_image(net_input[0])
        if FLAGS.transform_inverse_reg:
            net_input_jitter = tf.concat(net_input_jitter, axis=3)
            inputs['jformattedInput'] = self.deprocess_image(net_input_jitter[0])

        if FLAGS.dry_run:
            with tf.Session() as sess:
                input = sess.run(inputs)

            if FLAGS.input_type != 'REALESTATE_PP':
                scene_id = input["scene_id"]
                image_id = input["image_id"]
                print('Scene id:', scene_id)
                print('Image id:', image_id)

            intrinsics = input["intrinsics"]
            src_pose = input["src_pose"]
            ref_pose = input["ref_pose"]
            tgt_pose = input["tgt_pose"]

            tgt_img = input["tgt_image"][0]
            src_img = input["src_image"][0]
            ref_img = input["ref_image"][0]

            tgt = tgt_img * 255.
            src = src_img * 255.
            ref = ref_img * 255.
            tgt = tgt.astype(np.uint8)
            write_image(dryrun_dir + "/tgt.png",tgt)
            src = src.astype(np.uint8)
            write_image(dryrun_dir + "/src.png",src)
            ref = ref.astype(np.uint8)
            write_image(dryrun_dir + "/ref.png",ref)
            if 'hrestgt' in FLAGS.supervision:
                hres_tgt_img = input["hres_tgt_image"][0]
                hres_src_img = input["hres_src_image"][0]
                hres_ref_img = input["hres_ref_image"][0]
                hres_tgt = hres_tgt_img * 255.
                hres_src = hres_src_img * 255.
                hres_ref = hres_ref_img * 255.
                hres_tgt = hres_tgt.astype(np.uint8)
                write_image(dryrun_dir + "/hres_tgt.png",hres_tgt)
                hres_src = hres_src.astype(np.uint8)
                write_image(dryrun_dir + "/hres_src.png",hres_src)
                hres_ref = hres_ref.astype(np.uint8)
                write_image(dryrun_dir + "/hres_ref.png",hres_ref)

            # write out the PSVs'
            for i in range(FLAGS.num_psv_planes * 2):
                #test out reshape operation
                input['formattedInput'] = np.reshape(input['formattedInput'], (320,640,-1,3))
                write_image(dryrun_dir + "/formatInputReshaped_%s.png" % i, input['formattedInput'][:,:,i,:])
                #write_image(dryrun_dir + "/formatInput_%s.png" % i, input['formattedInput'][:,:,i*3:(i+1)*3])
                if FLAGS.transform_inverse_reg:
                    write_image(dryrun_dir + "/formatJitteredInput_%s.png" % i, input['jformattedInput'][:,:,i*3:(i+1)*3])
            exit()
        elif FLAGS.dry_run_inference:
            saver = tf.train.Saver([var for var in tf.trainable_variables()])
            ckpt_dir = FLAGS.checkpoint_dir
            if not os.path.exists(dryrun_dir):
                os.mkdir(dryrun_dir)
            print("Retrieving checkpoint from ", ckpt_dir)

            ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
            sv = tf.train.Supervisor(logdir=ckpt_dir, saver=None)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            prev_msi_np = np.zeros((1, 320, 640, FLAGS.num_msi_planes, 4))
            orig_pose_np = np.zeros((1, 4, 4))
            raw_tgt_image_np = np.zeros((1, 320, 640, 3))
            first_tgt_pose_np = np.zeros((1,3))
            with sv.managed_session(config=config) as sess:
                saver.restore(sess, ckpt_file)
                if FLAGS.transform_inverse_reg:
                    [input, rgba, jitter_rgba] = sess.run([inputs, rgba_layers, rgba_layers_jitter])
                else:
                    [input, rgba, tl0, mbw] = sess.run([inputs, rgba_layers, total_loss, msi_blend_weights], feed_dict={'setup/prev_msi:0': prev_msi_np, 'setup/view:0': 0, 'setup/orig_pose:0': orig_pose_np, 'setup/first_tgt:0': raw_tgt_image_np, 'setup/first_tgt_pose:0': first_tgt_pose_np})
                    orig_pose_0 = input["ref_pose"]
                    raw_tgt_image_0 = input["tgt_image"]
                    tgt_pose_0 = input["tgt_pose"]
                    [input1, rgba1, tl1, mbw1] = sess.run([inputs, rgba_layers, total_loss, msi_blend_weights], feed_dict={'setup/prev_msi:0': rgba, 'setup/view:0': 1, 'setup/orig_pose:0': orig_pose_0, 'setup/first_tgt:0': raw_tgt_image_0, 'setup/first_tgt_pose:0': tgt_pose_0})
                    [input2, rgba2, tl2, mbw2] = sess.run([inputs, rgba_layers, total_loss, msi_blend_weights], feed_dict={'setup/prev_msi:0': rgba1, 'setup/view:0': 2, 'setup/orig_pose:0': orig_pose_0, 'setup/first_tgt:0': raw_tgt_image_0, 'setup/first_tgt_pose:0': tgt_pose_0})

            print("Total Loss 0: ", tl0)
            print("Total Loss 1: ", tl1)
            print("Total Loss 2: ", tl2)
            for i in range(FLAGS.num_psv_planes * 2):
                write_image(dryrun_dir + "/formatInput_%s.png" % i, input['formattedInput'][:,:,i*3:(i+1)*3])
                if FLAGS.transform_inverse_reg:
                    write_image(dryrun_dir + "/formatJitteredInput_%s.png" % i, input['jformattedInput'][:,:,i*3:(i+1)*3])

            for i in range(FLAGS.num_psv_planes * 2):
                write_image(dryrun_dir + "/formatInput_%s_1.png" % i, input1['formattedInput'][:,:,i*3:(i+1)*3])
            
            for i in range(FLAGS.num_psv_planes * 2):
                write_image(dryrun_dir + "/formatInput_%s_2.png" % i, input2['formattedInput'][:,:,i*3:(i+1)*3])

            scene_id = input["scene_id"]
            image_id = input["image_id"]
            intrinsics = input["intrinsics"]
            src_pose = input["src_pose"]
            ref_pose = input["ref_pose"]
            tgt_pose = input["tgt_pose"]
            print('Scene id:', scene_id)
            print('Image id:', image_id)
            print('Scene id 1:', input1["scene_id"])
            print('Scene id 2:', input2["scene_id"])

            tgt_img = input["tgt_image"][0]
            src_img = input["src_image"][0]
            ref_img = input["ref_image"][0]
            src_img1 = input1["src_image"][0]
            ref_img1 = input1["ref_image"][0]
            src_img2 = input2["src_image"][0]
            ref_img2 = input2["ref_image"][0]

            tgt = tgt_img * 255.
            src = src_img * 255.
            ref = ref_img * 255.
            src1 = src_img1 * 255.
            ref1 = ref_img1 * 255.
            src2 = src_img2 * 255.
            ref2 = ref_img2 * 255.

            tgt = tgt.astype(np.uint8)
            write_image(dryrun_dir + "/tgt.png",tgt)
            src = src.astype(np.uint8)
            write_image(dryrun_dir + "/src"+ str(0) + ".png",src)
            ref = ref.astype(np.uint8)
            write_image(dryrun_dir + "/ref"+ str(0) + ".png",ref)

            src1 = src1.astype(np.uint8)
            write_image(dryrun_dir + "/src1"+ str(1) + ".png",src1)
            ref1 = ref1.astype(np.uint8)
            write_image(dryrun_dir + "/ref1"+ str(1) + ".png",ref1)

            src2 = src2.astype(np.uint8)
            write_image(dryrun_dir + "/src2"+ str(2) + ".png",src2)
            ref2 = ref2.astype(np.uint8)
            write_image(dryrun_dir + "/ref2"+ str(2) + ".png",ref2)

            if 'hrestgt' in FLAGS.supervision:
                hres_tgt_img = input["hres_tgt_image"][0]
                hres_src_img = input["hres_src_image"][0]
                hres_ref_img = input["hres_ref_image"][0]

                hres_tgt = hres_tgt_img * 255.
                hres_src = hres_src_img * 255.
                hres_ref = hres_ref_img * 255.
                hres_tgt = hres_tgt.astype(np.uint8)
                write_image(dryrun_dir + "/hres_tgt.png",hres_tgt)
                hres_src = hres_src.astype(np.uint8)
                write_image(dryrun_dir + "/hres_src.png",hres_src)
                hres_ref = hres_ref.astype(np.uint8)
                write_image(dryrun_dir + "/hres_ref.png",hres_ref)

            for i in range(FLAGS.num_msi_planes):
              alpha_img = rgba[0, :, :, i, 3] * 255.0
              rgb_img = (rgba[0, :, :, i, :3] + 1.) / 2. * 255
              msi_weights = mbw[0,:,:,i] * 255.0
              msi_weights = msi_weights.astype(int)
              write_image(dryrun_dir + '/msi_alpha_%.2d.png' % i, alpha_img)
              write_image(dryrun_dir + '/msi_rgb_%.2d.png' % i, rgb_img)
              write_image(dryrun_dir + '/msi_weights_%.2d.png' % i, msi_weights)
              if FLAGS.transform_inverse_reg:
                  jalpha_img = jitter_rgba[0, :, :, i, 3] * 255.0
                  jrgb_img = (jitter_rgba[0, :, :, i, :3] + 1.) / 2. * 255
                  write_image(dryrun_dir + '/jitter_msi_alpha_%.2d.png' % i, jalpha_img)
                  write_image(dryrun_dir + '/jitter_msi_rgb_%.2d.png' % i, jrgb_img)

            # reconstruct graph to fit re-rendering in memory
            tf.reset_default_graph()
            tgt_pos = tf.constant(input["tgt_pose"])
            intrinsics_inv = tf.matrix_inverse(tf.constant(intrinsics), name='intrinsics_inv')

            output_images = {}
            if FLAGS.input_type == 'ODS':
                output_images["output_image"] = self.deprocess_image(
                    self.msi_render_equirect_view(tf.constant(rgba), tf.expand_dims(tf.eye(4), axis=0), tgt_pos,
                                                  msi_planes, intrinsics))[0]
                output_images["output_depth"] = self.deprocess_image(
                    self.msi_render_equirect_depth(tf.constant(rgba), tf.expand_dims(tf.eye(4), axis=0), tgt_pos,
                                                   msi_planes, intrinsics))[0]
                output_images["output_pimage"] = self.deprocess_image(
                    self.msi_render_perspective_view(tf.constant(rgba), tf.expand_dims(tf.eye(4), axis=0), tgt_pos,
                                                     msi_planes, intrinsics))[0]

                output_images["output_image1"] = self.deprocess_image(
                    self.msi_render_equirect_view(tf.constant(rgba1), tf.expand_dims(tf.eye(4), axis=0), tgt_pos,
                                                  msi_planes, intrinsics))[0]
                output_images["output_depth1"] = self.deprocess_image(
                    self.msi_render_equirect_depth(tf.constant(rgba1), tf.expand_dims(tf.eye(4), axis=0), tgt_pos,
                                                   msi_planes, intrinsics))[0]
                output_images["output_pimage1"] = self.deprocess_image(
                    self.msi_render_perspective_view(tf.constant(rgba1), tf.expand_dims(tf.eye(4), axis=0), tgt_pos,
                                                     msi_planes, intrinsics))[0]

                output_images["output_image2"] = self.deprocess_image(
                    self.msi_render_equirect_view(tf.constant(rgba2), tf.expand_dims(tf.eye(4), axis=0), tgt_pos,
                                                  msi_planes, intrinsics))[0]
                output_images["output_depth2"] = self.deprocess_image(
                    self.msi_render_equirect_depth(tf.constant(rgba2), tf.expand_dims(tf.eye(4), axis=0), tgt_pos,
                                                   msi_planes, intrinsics))[0]
                output_images["output_pimage2"] = self.deprocess_image(
                    self.msi_render_perspective_view(tf.constant(rgba2), tf.expand_dims(tf.eye(4), axis=0), tgt_pos,
                                                     msi_planes, intrinsics))[0]
                if FLAGS.transform_inverse_reg:
                    jitter_pose = tf.constant(input["jitter_pose"])
                    jitter_pose_inv = tf.constant(input["jitter_pose_inv"])
                    tgt_pose_rt = tf.constant(input["tgt_pose_rt"])
                    tgt_pose_rt = tf.matmul(jitter_pose, tgt_pose_rt)
                    output_images["jitter_output_image"] = self.deprocess_image(self.msi_render_equirect_view(tf.constant(jitter_rgba), jitter_pose, tgt_pos, msi_planes, intrinsics))[0]
                    output_images["jitter_output_depth"] = self.deprocess_image(self.msi_render_equirect_depth(tf.constant(jitter_rgba), jitter_pose, tgt_pos, msi_planes, intrinsics))[0]
                    if 'src' in FLAGS.supervision:
                        output_images["jsrc_output_image"] = self.deprocess_image(self.msi_render_ods_view(tf.constant(jitter_rgba), -1, jitter_pose, tgt_pos, msi_planes, intrinsics))[0]
                    if 'ref' in FLAGS.supervision:
                        output_images["jref_output_image"] = self.deprocess_image(self.msi_render_ods_view(tf.constant(jitter_rgba), 1, jitter_pose, tgt_pos, msi_planes, intrinsics))[0]
                if 'src' in FLAGS.supervision:
                    output_images["src_output_image"] = self.deprocess_image(self.msi_render_ods_view(tf.constant(rgba), -1, tf.expand_dims(tf.eye(4), axis=0), tgt_pose, msi_planes, intrinsics))[0]
                if 'ref' in FLAGS.supervision:
                    output_images["ref_output_image"] = self.deprocess_image(self.msi_render_ods_view(tf.constant(rgba), 1, tf.expand_dims(tf.eye(4), axis=0), tgt_pose, msi_planes, intrinsics))[0]
            elif FLAGS.input_type == 'PP':
                output_images["output_image"] = self.deprocess_image(
                    self.mpi_render_view(tf.constant(rgba), tgt_pos, msi_planes, tf.constant(intrinsics)))[0]

            with tf.Session() as sess:
                outputs = sess.run(output_images)

            if FLAGS.input_type == 'ODS':
                write_image(dryrun_dir +"/tgtp_rendered.png", outputs["output_pimage"])
                write_image(dryrun_dir +"/tgt_rendered.png", outputs["output_image"])
                write_image(dryrun_dir +"/depth_rendered.png",outputs["output_depth"])

                write_image(dryrun_dir +"/tgtp_rendered_1.png", outputs["output_pimage1"])
                write_image(dryrun_dir +"/tgt_rendered_1.png", outputs["output_image1"])
                write_image(dryrun_dir +"/depth_rendered_1.png",outputs["output_depth1"])

                write_image(dryrun_dir +"/tgtp_rendered_2.png", outputs["output_pimage2"])
                write_image(dryrun_dir +"/tgt_rendered_2.png", outputs["output_image2"])
                write_image(dryrun_dir +"/depth_rendered_2.png",outputs["output_depth2"])

                if FLAGS.transform_inverse_reg:
                    if 'tgt' in FLAGS.supervision:
                        write_image(dryrun_dir +"/tgt_rendered_from_jitter.png", outputs["jitter_output_image"])
                        write_image(dryrun_dir +"/depth_rendered_from_jitter.png",outputs["jitter_output_depth"])
                    if 'src' in FLAGS.supervision:
                        write_image(dryrun_dir +"/src_rendered_from_jitter.png", outputs["jsrc_output_image"])
                    if 'ref' in FLAGS.supervision:
                        write_image(dryrun_dir +"/ref_rendered_from_jitter.png", outputs["jref_output_image"])
                if 'src' in FLAGS.supervision:
                    write_image(dryrun_dir +"/src_rendered.png", outputs["src_output_image"])
                if 'ref' in FLAGS.supervision:
                    write_image(dryrun_dir +"/ref_rendered.png", outputs["ref_output_image"])
            elif FLAGS.input_type == 'PP':
                write_image(dryrun_dir + "/tgt_rendered.png", outputs["output_image"])
            exit()

    return train_op, rgba_layers, orig_pose, first_tgt, first_tgt_pose

  def train(self, train_op, rgba_layers, orig_pose, first_tgt, first_tgt_pose, checkpoint_dir, continue_train, summary_freq,
            save_latest_freq, max_steps):
    """Runs the training procedure.
    Args:
      train_op: op for training the network
      rgba_layers: layers of the inferred MSI
      orig_pose: pose of camera for first input in sequence
      first_tgt: target image for first input in sequence
      first_tgt_pose: target pose for first input in sequence
      checkpoint_dir: where to save the checkpoints and summaries
      continue_train: whether to restore training from previous checkpoint
      summary_freq: summary frequency
      save_latest_freq: Frequency of model saving (overwrites old one)
      max_steps: maximum training steps
    """
    parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
    global_step = tf.Variable(0, name='global_step', trainable=False)
    incr_global_step = tf.assign(global_step, global_step + 1)
    saver = tf.train.Saver([var for var in tf.trainable_variables()]+ [global_step], max_to_keep=10)

    sv = tf.train.Supervisor(logdir=checkpoint_dir, save_summaries_secs=0, saver=None)
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 1
    with sv.managed_session(config=config) as sess:
      tf.logging.info('Trainable variables: ')
      for var in tf.trainable_variables():
        tf.logging.info(var.name)
      tf.logging.info('parameter_count = %d' % sess.run(parameter_count))
      if continue_train:
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if checkpoint is not None:
          tf.logging.info('Resume training from previous checkpoint')
          saver.restore(sess, checkpoint)
        else:
          tf.logging.info('Could not resume training from previous checkpoints')

      for step in range(1, max_steps):
        start_time = time.time()

        prev_msi = np.zeros((1, 320, 640, FLAGS.num_msi_planes, 4))
        orig_pose_np = np.zeros((1, 4, 4))
        raw_tgt_image_np = np.zeros((1, 320, 640, 3))
        first_tgt_pose_np = np.zeros((1,3))

        for i in range(FLAGS.num_synth + 1):
          fetches = {
            'train': train_op,
            'msi': rgba_layers,
            'orig_pose': orig_pose,
            'first_tgt': first_tgt,
            'first_tgt_pose': first_tgt_pose,
            'global_step': global_step,
            'incr_global_step': incr_global_step
          }
          if step % 10 == 0:
            fetches['summary'] = sv.summary_op
          
          results = sess.run(fetches, feed_dict={'setup/prev_msi:0': prev_msi, 'setup/view:0': i, 'setup/orig_pose:0': orig_pose_np, 'setup/first_tgt:0': raw_tgt_image_np, 'setup/first_tgt_pose:0': first_tgt_pose_np})
          prev_msi = results['msi']
          orig_pose_np = results['orig_pose']
          raw_tgt_image_np = results['first_tgt']
          first_tgt_pose_np = results['first_tgt_pose']

          gs = results['global_step']
          if step % summary_freq == 0:
            sv.summary_writer.add_summary(results['summary'], gs)
            tf.logging.info(
                '[Step %.8d] time: %4.4f/it' % (gs, time.time() - start_time))

        if step % save_latest_freq == 0:
          tf.logging.info(' [*] Saving checkpoint to %s...' % checkpoint_dir)
          saver.save(sess, os.path.join(checkpoint_dir, 'model.latest'), global_step=global_step)

  def format_realestate_network_input(self, ref_image, src_image, ref_pose,
                                      src_pose, planes, intrinsics):
    """Format the network input (reference source image + PSV of the 2nd image).

    Args:
      ref_image: reference source image [batch, height, width, 3]
      psv_src_images: stack of source images (excluding the ref image)
                      [batch, height, width, 3*(num_source -1)]
      ref_pose: reference world-to-camera pose (where PSV is constructed)
                [batch, 4, 4]
      psv_src_poses: input poses (world to camera) [batch, num_source-1, 4, 4]
      planes: list of scalar depth values for each plane
      intrinsics: camera intrinsics [batch, 3, 3]
    Returns:
      net_input: [batch, height, width, (num_source-1)*#planes*3 + 3]
    """
    
    psv_src_images = tf.concat([ref_image, src_image], axis=-1)
    psv_src_poses = tf.stack([ref_pose, src_pose], axis=1)
    _, num_psv_source, _, _ = psv_src_poses.get_shape().as_list()

    net_input = []
    net_input.append(ref_image)
    for i in range(num_psv_source):
      ref_pose_inv = tf.matrix_inverse(ref_pose)
      if FLAGS.jitter:
        jitter_pose_inv = tf.get_default_graph().get_tensor_by_name("jitter_pose_inv:0")
        ref_pose_inv = tf.matmul(tf.matrix_inverse(ref_pose), jitter_pose_inv)

      curr_pose = tf.matmul(psv_src_poses[:, i], ref_pose_inv)
      curr_image = psv_src_images[:, :, :, i * 3:(i + 1) * 3]
      curr_psv = pj.plane_sweep(curr_image, planes, curr_pose, intrinsics)
      net_input.append(curr_psv)

    net_input = tf.concat(net_input, axis=3)
    return net_input

  def format_gcn_network_input(self, ref_image, src_image,
                           ref_pose, src_pose, planes, intrinsics):
    """Format the network input into double psv for gcn training.
    Args:
      ref_image: reference image [batch, height, width, 3]
      src_image: source image [batch, height, width, 3]
      ref_pose: reference world-to-camera pose [batch, 4, 4]
      src_pose: src world-to-camera [batch, 4, 4]
      planes: list of scalar depth values for each plane
      intrinsics: camera intrinsics [batch, 3, 3]
    Returns:
      net_input: [batch, height, width, num_source (2) * #planes * 3]
    """
    # stack the ref,src images for constructing the psv
    psv_src_images = tf.concat([ref_image, src_image], axis=-1)
    psv_src_poses = tf.concat([ref_pose, src_pose], axis=0)
    num_psv_source, _, _ = psv_src_poses.get_shape().as_list()

    coord = tf.get_default_graph().get_tensor_by_name('coord:0')
    ref_pose_inv = tf.get_default_graph().get_tensor_by_name("ref_pose_inv:0")

    # Plane sweep interpolated from psv source images
    net_input = []
    for i in range(num_psv_source):
      curr_pose = tf.matmul(psv_src_poses[i:i+1, :, :], ref_pose_inv)
      curr_image = psv_src_images[:, :, :, i * 3:(i + 1) * 3]
      curr_psv = pj.gcn_sphere_sweep(curr_image, -1 if (i % 2) == 0 else 1, planes,
                                     coord, curr_pose, intrinsics)
      net_input.append(curr_psv)

    net_input = tf.concat(net_input, axis=3)
    return net_input[0][0]

  def format_network_input(self, ref_image, src_image,
                           ref_pose, src_pose, planes, intrinsics):
    """Format the network input into double psv.
    Args:
      ref_image: reference image [batch, height, width, 3]
      src_image: source image [batch, height, width, 3]
      ref_pose: reference world-to-camera pose [batch, 4, 4]
      src_pose: src world-to-camera [batch, 4, 4]
      planes: list of scalar depth values for each plane
      intrinsics: camera intrinsics [batch, 3, 3]
    Returns:
      net_input: [batch, height, width, num_source (2) * #planes * 3]
    """
    # stack the ref,src images for constructing the psv
    psv_src_images = tf.concat([ref_image, src_image], axis=-1)
    psv_src_poses = tf.concat([ref_pose, src_pose], axis=0)
    num_psv_source, _, _ = psv_src_poses.get_shape().as_list()

    if FLAGS.input_type != 'ODS':
        ref_pose_inv = tf.get_default_graph().get_tensor_by_name("interp_pose_inv:0")
    else:
        ref_pose_inv = tf.get_default_graph().get_tensor_by_name("ref_pose_inv:0")

    # Jitter pose for transform-inverse training
    if FLAGS.jitter:
        jitter_pose_inv = tf.get_default_graph().get_tensor_by_name("jitter_pose_inv:0")
        ref_pose_inv = tf.matmul(ref_pose_inv, jitter_pose_inv)

    # Plane sweep interpolated from psv source images
    net_input = []
    for i in range(num_psv_source):
      curr_pose = tf.matmul(psv_src_poses[i:i+1, :, :], ref_pose_inv)
      curr_image = psv_src_images[:, :, :, i * 3:(i + 1) * 3]
      curr_psv = self.sweep_src(curr_image, -1 if (i % 2) == 0 else 1, planes, curr_pose, intrinsics)
      net_input.append(curr_psv)
    net_input = tf.concat(net_input, axis=3)
    return net_input

  def format_network_input_synth(self, ref_image, src_image, tgt_pose,
                           ref_pose, src_pose, planes, intrinsics, rgba_layers=None):
    """Format the network input into double psv.
    Args:
      ref_image: reference image [batch, height, width, 3]
      src_image: source image [batch, height, width, 3
      tgt_pose: target pose for first input in sequence
      ref_pose: reference world-to-camera pose [batch, 4, 4]
      src_pose: src world-to-camera [batch, 4, 4]
      planes: list of scalar depth values for each plane
      intrinsics: camera intrinsics [batch, 3, 3]
      rgba_layers: MSI from previous frame
    Returns:
      net_input: [batch, height, width, num_source (2) * #planes * 3]
    """
    psv_src_images = tf.concat([ref_image, src_image], axis=-1)
    psv_src_poses = tf.concat([tgt_pose, tgt_pose], axis=0)
    num_psv_source, _, _= psv_src_poses.get_shape().as_list()

    if FLAGS.input_type != 'ODS':
        ref_pose_inv = tf.get_default_graph().get_tensor_by_name("interp_pose_inv:0")
    else:
        ref_pose_inv = tf.linalg.inv(ref_pose)
    # Jitter pose for transform-inverse training
    if FLAGS.jitter:
        jitter_pose_inv = tf.get_default_graph().get_tensor_by_name("jitter_pose_inv:0")
        ref_pose_inv = tf.matmul(ref_pose_inv, jitter_pose_inv)

    net_input = []

    for i in range(num_psv_source):
      curr_pose = tf.matmul(ref_pose_inv, psv_src_poses[i:i+1, :, :])
      curr_image = psv_src_images[:, :, :, i * 3:(i + 1) * 3]
      curr_psv = self.sweep_src(curr_image, -1 if (i % 2) == 0 else 1, planes, curr_pose, intrinsics)
      net_input.append(curr_psv)
    net_input = tf.concat(net_input, axis=3)
    return net_input

  def create_spherical_weights(self, epsilon = 1.0e-12):
      width = FLAGS.width
      height = FLAGS.height

      grid = tf.meshgrid(tf.linspace(-np.pi + epsilon  , np.pi + epsilon , width),
            tf.linspace(-np.pi / 2.0 + epsilon, np.pi / 2.0 + epsilon , height))
      delta = np.pi / height
      grid_shift = tf.meshgrid(tf.linspace(-np.pi + delta , np.pi + delta , width),
                  tf.linspace(-np.pi / 2.0 + delta / 2.0, np.pi / 2.0 + delta / 2.0 , height))

      spherical_attention = 1 / abs(tf.cos(grid[1])-tf.cos(grid_shift[1])) * abs(grid_shift[0] - grid[0])
      return spherical_attention

  def sweep_ref(self, image, intrinsics):
    #TODO: is this used in the cleanup version? i.e. should the cleanup version support real-estate training?
    batch, _, _, _ = image.get_shape().as_list()
    if FLAGS.jitter:
      pose = tf.get_default_graph().get_tensor_by_name("jitter_pose_inv:0")
    else:
      pose = tf.tile(tf.expand_dims(tf.eye(4, dtype=tf.float32), 0), [batch, 1, 1])
    if FLAGS.input_type == 'ODS':
      return pj.ods_centered_sphere_sweep(image, 0, [1], pose, intrinsics)
    else:
      return self.sweep_src(image, 0, [1], pose, intrinsics)

  def sweep_src(self, image, order, depths, pose, intrinsics):
    if FLAGS.input_type == 'ODS':
        return pj.ods_sphere_sweep(image, order, depths, pose, intrinsics)
    else:
        return pj.perspective_plane_sweep(image, order, depths, pose, intrinsics)

  def preprocess_image(self, image):
    """Preprocess the image for CNN input.
    Args:
      image: the input image in either float [0, 1] or uint8 [0, 255]
    Returns:
      A new image converted to float with range [-1, 1]
    """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image * 2 - 1

  def deprocess_image(self, image):
    """Undo the preprocessing.
    Args:
      image: the input image in float with range [-1, 1]
    Returns:
      A new image converted to uint8 [0, 255]
    """
    image = (image + 1.) / 2.
    return tf.image.convert_image_dtype(image, dtype=tf.uint8)

  def preprocess_depth(self,image):
    return tf.image.convert_image_dtype(image, dtype=tf.float32)

  def deprocess_depth_image(self, image):
      #TODO: why not + 1. / 2. here?
    """Undo the preprocessing.
    Args:
      image: the input image in float with range [-1, 1]
    Returns:
      A new image converted to uint8 [0, 255]
    """
    return tf.image.convert_image_dtype(image, dtype=tf.uint8)

  def inv_depths(self, start_depth, end_depth, num_depths):
    """Sample reversed, sorted inverse depths between a near and far plane.

    Args:
      start_depth: The first depth (i.e. near plane distance).
      end_depth: The last depth (i.e. far plane distance).
      num_depths: The total number of depths to create. start_depth and
          end_depth are always included and other depths are sampled
          between them uniformly according to inverse depth.
    Returns:
      The depths sorted in descending order (so furthest first). This order is
      useful for back to front compositing.
    """
    inv_start_depth = 1.0 / start_depth
    inv_end_depth = 1.0 / end_depth
    depths = [start_depth, end_depth]
    for i in range(1, num_depths - 1):
      fraction = float(i) / float(num_depths - 1)
      inv_depth = inv_start_depth + (inv_end_depth - inv_start_depth) * fraction
      depths.append(1.0 / inv_depth)
    depths = sorted(depths)
    return depths[::-1]

