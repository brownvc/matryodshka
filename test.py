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

"""Main script for evaluating the model on a test set.
"""
from __future__ import division
import os
import tensorflow as tf
print(tf.__version__)
import numpy as np
from matryodshka.msi import MSI
from matryodshka.data_loader import ReplicaSequenceDataLoader
from matryodshka.utils import write_image, load_mesh_input
import glob
from tqdm import tqdm
from geometry.spherical import tf_random_rotation

# Note that the flags below are a subset of all flags. The remainder (data
# loading relevant) are defined in loader.py.
flags = tf.app.flags

# i/o
flags.DEFINE_string('cameras_glob', 'glob/test/regular/*.txt',
                    'Glob string for test set camera files.')
flags.DEFINE_string('image_dir', '/path/to/test_640x320', 'Path to testing image directories.')
flags.DEFINE_string('hres_image_dir', '/path/to/test_4096x2048', 'Path to high-resolution testing image directories.')
flags.DEFINE_string('output_root', './test', 'Root of directory to write test results.')
flags.DEFINE_string('checkpoint_dir', 'checkpoints',
                    'Location to load the models.')
flags.DEFINE_string('experiment_name', '', 'Name for the experiment model to test.')
flags.DEFINE_integer('shuffle_seq_length', 3, 'Number of images for each input group.')

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
flags.DEFINE_integer('random_seed', 8964, 'Random seed.')

# for model export
flags.DEFINE_boolean('net_only', False, 'Extract only the network')
flags.DEFINE_boolean('smoothed', False, 'Smooth conv2d transpose ops')

# experiments flags
flags.DEFINE_string('supervision', 'tgt', "Images to supervise on. [tgt, ref, src, hrestgt] concatenated with _")
flags.DEFINE_float('rot_factor',1.0,'for experiments with rotation jittering range')
flags.DEFINE_float('tr_factor',1.0,'for experiments with translation jittering range')

# test script specific 
flags.DEFINE_string('test_type', '',
                    'Type of testings, can concatenate with _ : [on_video, high_res, high_res_only]')
flags.DEFINE_string('prefix', '', 'Prefix for saving results, used for generating supplement materials.')
flags.DEFINE_string('test_outputs', 'rgba_layers_src_image_ref_image_tgt_image_blend_weights_alphas',
    'Which outputs to save. Can concat the following with "_": '
    '[src_image, ref_image, tgt_image, psp (for perspective crop), hres_tgt_image, src_output_image, ref_output_image, psv, alphas, blend_weights, rgba_layers].')
flags.DEFINE_integer('num_runs', -1, 'Custom number of runs.')

flags.DEFINE_boolean('gcn',False,'Train with gcn.')
flags.DEFINE_integer('subdiv',7,'subdivision level for the spherical mesh we want to operate on.')

FLAGS = flags.FLAGS

def main(_):

  assert FLAGS.batch_size == 1, 'Currently, batch_size must be 1 when testing.'

  #get number of iterations if not specified by user, i.e find the number of samples listed in the txt files
  if FLAGS.num_runs < 0: #no specified number of runs
    files = glob.glob(FLAGS.cameras_glob)
    num_runs = 0
    for file in files:
        f = open(file)
        f_content = f.read()
        f_lines = f_content.split("\n")
        num_runs += len(f_lines)-1
    FLAGS.num_runs = num_runs

  #run low-res inference
  if 'high_res_only' not in FLAGS.test_type:
      #load test data
      assert 'hrestgt' not in FLAGS.supervision #to not load in high res images
      data_loader = ReplicaSequenceDataLoader(FLAGS.cameras_glob, FLAGS.image_dir, FLAGS.hres_image_dir, False,
                               FLAGS.shuffle_seq_length, FLAGS.random_seed)
      inputs = data_loader.sample_batch()

      #add additional needed tensors
      inputs['ref_pose_inv'] = tf.matrix_inverse(inputs['ref_pose'], name='ref_pose_inv')
      inputs['jitter_pose'] = tf.identity(tf_random_rotation(FLAGS.rot_factor, FLAGS.tr_factor), name='jitter_pose')
      inputs['jitter_pose_inv'] = tf.matrix_inverse(inputs['jitter_pose'], name='jitter_pose_inv')
      raw_hres_ref_image = None
      raw_hres_src_image = None

      #for gcn
      if FLAGS.gcn:
          coord, support, pix2vertex_lookup = load_mesh_input()
          inputs['p2v'] = tf.dtypes.cast(tf.identity(pix2vertex_lookup, name='p2v'), dtype=tf.float32)
          inputs['coord'] = tf.identity(coord, name='coord')
          inputs['support'] = [tf.SparseTensor(indices=np.asarray(support[i][0], dtype=np.float32),
                                               values=np.asarray(support[i][1], dtype=np.float32),
                                               dense_shape=np.asarray(support[i][2], dtype=np.float32)) for i in range(len(support))]

      #infer msi
      model = MSI()
      psv_planes = model.inv_depths(FLAGS.min_depth, FLAGS.max_depth, FLAGS.num_psv_planes)
      msi_planes = model.inv_depths(FLAGS.min_depth, FLAGS.max_depth, FLAGS.num_msi_planes)
      if FLAGS.gcn:
          with tf.name_scope('inference'):
              outputs, net_input = model.infer_gcn_msi(inputs['src_image'], inputs['ref_image'], inputs['ref_pose'],
                                                       inputs['src_pose'], inputs['intrinsics'], FLAGS.which_color_pred,
                                                       FLAGS.num_msi_planes, psv_planes, inputs, extra_outputs=FLAGS.test_outputs)
      else:
          assert FLAGS.jitter==False
          outputs, net_input = model.infer_msi(
              inputs['src_image'], inputs['ref_image'], raw_hres_src_image, raw_hres_ref_image, inputs['ref_pose'],
              inputs['src_pose'], inputs['intrinsics'], FLAGS.which_color_pred, FLAGS.num_msi_planes,
              psv_planes, FLAGS.test_outputs, ngf=FLAGS.ngf)
          if FLAGS.transform_inverse_reg:
              FLAGS.jitter = True
              jitter_outputs, jitter_net_input = model.infer_msi(
                inputs['src_image'], inputs['ref_image'], raw_hres_src_image, raw_hres_ref_image, inputs['ref_pose'],
                inputs['src_pose'], inputs['intrinsics'], FLAGS.which_color_pred,  FLAGS.num_msi_planes,
                psv_planes, FLAGS.test_outputs, ngf=FLAGS.ngf)
      
      #get re-rendering outputs as specified by FLAGS.test_outputs
      if 'tgt_image' in FLAGS.test_outputs:
          outputs['output_image'] = model.deprocess_image(model.msi_render_equirect_view(outputs['rgba_layers'],
                                                                                         tf.expand_dims(tf.eye(4), axis=0),
                                                                                         inputs['tgt_pose'],
                                                                                         msi_planes,
                                                                                         inputs['intrinsics']))
          outputs['output_depth'] = model.deprocess_depth_image(model.msi_render_equirect_depth(outputs['rgba_layers'],
                                                                                                tf.expand_dims(tf.eye(4), axis=0),
                                                                                                inputs['tgt_pose'],
                                                                                                msi_planes,
                                                                                                inputs['intrinsics']))
          if FLAGS.transform_inverse_reg:
              outputs['jitter_output_image'] = model.deprocess_image(model.msi_render_equirect_view(jitter_outputs['rgba_layers'],
                                                                                                    inputs["jitter_pose"],
                                                                                                    inputs['tgt_pose'],
                                                                                                    msi_planes,
                                                                                                    inputs['intrinsics']))
      if 'psp' in FLAGS.test_outputs:
          outputs['output_psp0'] = model.deprocess_image(model.msi_render_perspective_view(outputs['rgba_layers'],
                                                                                           tf.expand_dims(tf.eye(4), axis=0), inputs['tgt_pose'], msi_planes, inputs['intrinsics'], viewing_window=0))
          outputs['output_psp1'] = model.deprocess_image(model.msi_render_perspective_view(outputs['rgba_layers'],
                                                                                           tf.expand_dims(tf.eye(4), axis=0), inputs['tgt_pose'], msi_planes, inputs['intrinsics'], viewing_window=1))
          outputs['output_psp2'] = model.deprocess_image(model.msi_render_perspective_view(outputs['rgba_layers'],
                                                                                           tf.expand_dims(tf.eye(4), axis=0), inputs['tgt_pose'], msi_planes, inputs['intrinsics'], viewing_window=2))
          outputs['output_psp3'] = model.deprocess_image(model.msi_render_perspective_view(outputs['rgba_layers'],
                                                                                           tf.expand_dims(tf.eye(4), axis=0), inputs['tgt_pose'], msi_planes, inputs['intrinsics'], viewing_window=3))
      if 'src_output_image' in FLAGS.test_outputs:
          outputs['output_src'] = model.deprocess_image(model.msi_render_ods_view(outputs['rgba_layers'],
                                                                                  -1,
                                                                                  tf.expand_dims(tf.eye(4), axis=0),
                                                                                  inputs['tgt_pose'],
                                                                                  msi_planes,
                                                                                  inputs['intrinsics']))
      if 'ref_output_image' in FLAGS.test_outputs:
          outputs['output_ref'] = model.deprocess_image(model.msi_render_ods_view(outputs['rgba_layers'],
                                                                                  1,
                                                                                  tf.expand_dims(tf.eye(4), axis=0),
                                                                                  inputs['tgt_pose'],
                                                                                  msi_planes,
                                                                                  inputs['intrinsics']))

      # run inference and save relevant data
      global_step = tf.Variable(0, name='global_step', trainable=False)
      saver = tf.train.Saver([var for var in tf.trainable_variables()]+[global_step])
      ckpt_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.experiment_name)
      ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
      sv = tf.train.Supervisor(logdir=ckpt_dir, saver=None)
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      with sv.managed_session(config=config) as sess:
          for run in tqdm(range(FLAGS.num_runs)):
              #output directory name: [scene]_[1st src file]_[2nd src file]_[tgt file].
              tf.logging.info('Progress: %d/%d' % (run, FLAGS.num_runs))
              saver.restore(sess, ckpt_file)
              if FLAGS.transform_inverse_reg:
                  [ins, outs, jitter_outs, step] = sess.run([inputs, outputs, jitter_outputs, global_step])
              else:
                  [ins, outs, step] = sess.run([inputs, outputs, global_step])

              #get output dirs
              if 'on_video' in FLAGS.test_type:
                  dirname = 'video_'
                  if FLAGS.prefix != '': dirname += '%s_' % FLAGS.prefix
                  dirname += ins['scene_id'][0]
              else:
                  dirname = ins['scene_id'][0]
              dirname += '_%s' % ins['image_id'][0][0]
              dirname += '%s' % ins['image_id'][0][1]
              dirname += '%s' % ins['image_id'][0][2]

              output_dir = os.path.join(FLAGS.output_root, FLAGS.experiment_name, dirname)
              if not tf.gfile.IsDirectory(output_dir):
                  tf.gfile.MakeDirs(output_dir)
              print("Saving to %s" % output_dir)

              #save trained number of steps to local as well
              txt_output_dir = os.path.join(FLAGS.output_root, FLAGS.experiment_name)
              if run == 0:
                  with open(txt_output_dir + "/step.txt","w") as file:
                      file.write("%d" % step)
                  print("Saving the steps to %s" % output_dir);

              if 'tgt_image' in FLAGS.test_outputs:
                  write_image(output_dir + '/tgt_image_%s.png' % dirname, ins['tgt_image'][0] * 255.0)
                  write_image(output_dir + '/output_tgt_%s.png' % dirname, outs['output_image'][0])
                  write_image(output_dir + '/output_depth_%s.png' % dirname, outs['output_depth'][0])
                  if FLAGS.transform_inverse_reg:
                      write_image(output_dir + '/jitter_output_tgt_%s.png' % dirname, outs['jitter_output_image'][0])
                      write_image(output_dir + '/jitter_output_depth_%s.png' % dirname, outs['jitter_output_depth'][0])

              if 'ref_output_image' in FLAGS.test_outputs:
                  write_image(output_dir + '/output_ref_%s.png' % dirname, outs['output_ref'][0])
              if 'src_output_image' in FLAGS.test_outputs:
                  write_image(output_dir + '/output_src_%s.png' % dirname, outs['output_src'][0])
              
              if 'psp' in FLAGS.test_outputs:
                  write_image(output_dir + '/output_ptgt0_%s.png' % dirname, outs['output_psp0'][0])
                  write_image(output_dir + '/output_ptgt1_%s.png' % dirname, outs['output_psp1'][0])
                  write_image(output_dir + '/output_ptgt2_%s.png' % dirname, outs['output_psp2'][0])
                  write_image(output_dir + '/output_ptgt3_%s.png' % dirname, outs['output_psp3'][0])

              if 'src_image' in FLAGS.test_outputs:
                  write_image(output_dir + '/src_image_%s.png' % dirname, ins['src_image'][0] * 255.)

              if 'ref_image' in FLAGS.test_outputs:
                  write_image(output_dir + '/ref_image_%s.png' % dirname, ins['ref_image'][0] * 255.)

              if 'psv' in FLAGS.test_outputs:
                  for j in range(FLAGS.num_psv_planes):
                      plane_img = (outs['psv'][0, :, :, j * 3:(j + 1) * 3] + 1.) / 2. * 255
                      write_image(output_dir + '/psv_plane_%.3d.png' % j, plane_img)

              if 'blend' in FLAGS.which_color_pred and 'blend_weights' in FLAGS.test_outputs:
                  #save as .npy
                  np.save(output_dir + '/blend_weights.npy', outs['blend_weights'])
                  for i in range(FLAGS.num_msi_planes):
                      weight_img = outs['blend_weights'][0, :, :, i] * 255.0
                      write_image(output_dir + '/blend_weight_%.3d.png' % i, weight_img)

              if 'alphas' in FLAGS.test_outputs:
                  np.save(output_dir + '/alphas.npy', outs['alphas'])

              if 'rgba_layers' in FLAGS.test_outputs:
                  for i in range(FLAGS.num_msi_planes):
                      alpha_img = outs['rgba_layers'][0, :, :, i, 3] * 255.0
                      rgb_img = (outs['rgba_layers'][0, :, :, i, :3] + 1.) / 2. * 255
                      write_image(output_dir + '/msi_alpha_%.2d.png' % i, alpha_img)
                      write_image(output_dir + '/msi_rgb_%.2d.png' % i, rgb_img)
                      if FLAGS.transform_inverse_reg:
                          jalpha_img = jitter_outs['rgba_layers'][0, :, :, i, 3] * 255.0
                          jrgb_img = (jitter_outs['rgba_layers'][0, :, :, i, :3] + 1.) / 2. * 255
                          write_image(output_dir + '/jitter_msi_alpha_%.2d.png' % i, jalpha_img)
                          write_image(output_dir + '/jitter_msi_rgb_%.2d.png' % i, jrgb_img)

  # run high-res re-rendering
  if 'high_res' in FLAGS.test_type:
      # clear the graph
      tf.reset_default_graph()

      # load data
      FLAGS.supervision += 'hrestgt' #to enable high-res dataloading
      data_loader = ReplicaSequenceDataLoader(FLAGS.cameras_glob, FLAGS.image_dir, FLAGS.hres_image_dir, False,
                                              FLAGS.shuffle_seq_length, FLAGS.random_seed, repeat_sample=64) #TODO: not sure why it's 64 but not 32 but this is what works :)
      inputs = data_loader.sample_batch()

      # add additional input tensors
      inputs['ref_pose_inv'] = tf.matrix_inverse(inputs['ref_pose'], name='ref_pose_inv')
      inputs['jitter_pose'] = tf.identity(tf.expand_dims(tf_random_rotation(1,1),0), name='jitter_pose')
      inputs['jitter_pose_inv'] = tf.matrix_inverse(inputs['jitter_pose'], name='jitter_pose_inv')

      # get hres psv
      model = MSI()
      psv_planes = model.inv_depths(FLAGS.min_depth, FLAGS.max_depth, FLAGS.num_psv_planes)
      hres_ref_image = model.preprocess_image(inputs['hres_ref_image'])
      hres_src_image = model.preprocess_image(inputs['hres_src_image'])
      batch_size, hres_img_height, hres_img_width, _ = hres_src_image.get_shape().as_list()

      # build each psv plane separately to fit into memory, and over-composite the re-rendered output image as we do each psv
      plane_idx =  tf.placeholder(tf.int32) #placeholder for plane index
      plane_blend_weight = tf.placeholder(tf.float32, shape=(1,None,None,1))
      plane_alpha = tf.placeholder(tf.float32, shape=(1,None,None,1))

      curr_psv_plane = tf.slice(tf.constant(psv_planes),[plane_idx],[1])
      hres_net_input = model.format_network_input(hres_ref_image,
                                                  hres_src_image,
                                                  inputs['ref_pose'],
                                                  inputs['src_pose'],
                                                  curr_psv_plane, #psv_planes[],
                                                  inputs['intrinsics'])

      upsampled_blend_weights = tf.image.resize(plane_blend_weight,
                                                [FLAGS.hres_height, FLAGS.hres_width],
                                                align_corners=True,
                                                method=tf.image.ResizeMethod.BILINEAR)
      upsampled_alphas = tf.image.resize(plane_alpha,
                                         [FLAGS.hres_height, FLAGS.hres_width],
                                         align_corners=True, method=tf.image.ResizeMethod.BILINEAR)

      ufg_rgb = hres_net_input[:, :, :, 0:3]
      ubg_rgb = hres_net_input[:, :, :, 3:6]
      ucurr_alpha = upsampled_alphas
      uw = upsampled_blend_weights
      ucurr_rgb = uw * ufg_rgb + (1 - uw) * ubg_rgb
      ucurr_rgba = tf.concat([ucurr_rgb, ucurr_alpha], axis=3)
      urgba_layers = ucurr_rgba
      urgba_layers = tf.reshape(urgba_layers, [batch_size, hres_img_height, hres_img_width, 1, 4])

      hres_outputs = {}
      hres_outputs['hres_output_image'] = \
      model.msi_render_equirect_view_single(urgba_layers, tf.expand_dims(tf.eye(4), axis=0),
                                            inputs['tgt_pose'], curr_psv_plane,
                                            inputs['intrinsics'])

      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      with tf.Session(config=config) as sess:
          for run in tqdm(range(FLAGS.num_runs)):
              for i in range(FLAGS.num_psv_planes):
                  # get inputs
                  hres_ins = sess.run(inputs)

                  # get output dirs
                  if 'on_video' in FLAGS.test_type:
                      dirname = 'video_'
                      if FLAGS.prefix != '': dirname += '%s_' % FLAGS.prefix
                      dirname += hres_ins['scene_id'][0]
                  else:
                      dirname = hres_ins['scene_id'][0]
                  dirname += '_%s' % hres_ins['image_id'][0][0]
                  dirname += '%s' % hres_ins['image_id'][0][1]
                  dirname += '%s' % hres_ins['image_id'][0][2]
                  output_dir = os.path.join(FLAGS.output_root, FLAGS.experiment_name, dirname)
                  print("Rendering %s for the %dth plane." % (output_dir, i))

                  curr_blend_weights = np.load(output_dir + "/blend_weights.npy")
                  curr_alphas = np.load(output_dir + "/alphas.npy")
                  curr_blend_weight = curr_blend_weights[:,:,:,i:i+1]
                  curr_alpha = curr_alphas[:,:,:,i:i+1]
                  hres_outs = sess.run(hres_outputs, feed_dict={plane_idx: i,
                                                                plane_blend_weight: curr_blend_weight,
                                                                plane_alpha: curr_alpha})

                  cur_hres_output = hres_outs['hres_output_image'][0].astype(np.float32)

                  # over-composite rgb (numpy)
                  rgb = cur_hres_output[:, :, :, :3]
                  alpha = cur_hres_output[:, :, :, 3:]
                  alpha_as_depth = np.tile(cur_hres_output[:,:,:,3:], (1,1,1,3))
                  if i == 0:
                      hres_output = rgb
                      hres_depth = 0.
                  else:
                      hres_output = hres_output * (1. - alpha) + rgb * alpha
                      hres_depth = (i / FLAGS.num_psv_planes) * alpha_as_depth + hres_depth * (1.0 - alpha_as_depth)

              # deprocess
              hres_output = ((hres_output + 1.) / 2.)
              hres_output = np.squeeze(hres_output * 255.)
              hres_depth = np.squeeze(hres_depth * 255.)
              if not tf.gfile.IsDirectory(output_dir):
                  tf.gfile.MakeDirs(output_dir)

              # write hres output
              print("Saving high-res output to %s" % output_dir)
              write_image(output_dir + '/output_hrestgt_%s.png' % dirname, hres_output)
              write_image(output_dir + '/output_hresdepth_%s.png' % dirname, hres_depth)
if __name__ == '__main__':
    tf.app.run()
