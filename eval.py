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
Quantitative evaluation of view synthesis results.
This script compares data and dumps scores to a JSON file.
"""

import os
import json
import numpy as np
import PIL.Image as pil
from multiprocessing.dummy import Pool
import tensorflow as tf
from tensorflow import app
import elpips.elpips as elpips
import math

flags = tf.app.flags
flags.DEFINE_string('result_root', './test',
                    'Root directory for writing results.')
flags.DEFINE_string('model_name', 'ods-wotemp-elpips-coord',
                    'Name of model to evaluate.')
flags.DEFINE_string('output_table', './test/results/ods-wotemp-elpips-coord.json',
                    'Filename for writing the output.')
flags.DEFINE_boolean('remove_pole',False,'To evaluate without considering pole area, i.e. top and bottom quarter of rows of pixels .')
flags.DEFINE_string('videos', 'room_0 room_2 office_0 apartment_0', 'Videos that we want to evaluate on, separated by space.')

#evaluation-related 
flags.DEFINE_string('eval_type','','Type of testings, can concatenate with _ : [on_video]')

FLAGS = flags.FLAGS

def load_image(imfile):
  fh = tf.gfile.GFile(imfile, 'r')
  raw_im = pil.open(fh)
  result = np.array(raw_im, dtype=np.float32)
  if len(np.shape(result)) != 3:
      result = np.expand_dims(result, axis=-1)
      result = np.tile(result, [1,1,3])
  h,w,ch = result.shape
  if FLAGS.remove_pole:
      result = result[h/4:3*h/4,:,:]
  return result

def collect_examples(result_root, model_names):
  """Find non-video test examples that exist for all models."""
  counts = {}
  for model_name in model_names:
    examples = os.listdir(os.path.join(result_root, model_name))
    for e in examples:
        if e.endswith(".txt"):
            #skip the step.txt file
            continue
        if 'video' in e:
            #skip the video frame test samples
            continue
        counts[e] = counts.get(e, 0) + 1
  result = [k for k, v in counts.items() if v == len(model_names)]
  skipped = [k for k, v in counts.items() if v != len(model_names)]
  assert not skipped
  return result

def collect_video_examples(result_root, model_names, scene_names):
  """Find video test examples that exist for all models."""
  results = []
  for model_name in model_names:
    examples = os.listdir(os.path.join(result_root, model_name))
    for scene in scene_names:
        counts = {}
        for e in examples:
            if e.endswith(".txt") or 'video' not in e:
                continue
            if scene in e:
                counts[e] = counts.get(e, 0) + 1

        result = [k for k, v in counts.items() if v == len(model_names)]
        skipped = [k for k, v in counts.items() if v != len(model_names)]
        assert not skipped
        results.append(result)

  assert(len(results) == len(scene_names))
  return np.stack(results,axis=0)

def collect_video_consecutive_examples(result_root, model_names, scene_names):
    """Find examples that exist for all models in consecutive frame pairs."""
    results = []
    consecutive_results = []
    for model_name in model_names:
        examples = os.listdir(os.path.join(result_root, model_name))
        for scene in scene_names:
            counts = {}
            for e in examples:
                if e.endswith(".txt") or 'video' not in e:
                    continue
                if scene in e:
                    counts[e] = counts.get(e, 0) + 1
            result = [k for k, v in counts.items() if v == len(model_names)]
            skipped = [k for k, v in counts.items() if v != len(model_names)]
            assert not skipped
            results.append(result)

    assert len(results) == len(scene_names)
    for i in range(len(results)):
        #for each video sort by frame number
        results[i] = np.sort(results[i],axis=0).tolist()
        consecutive_result = [[results[i][j],results[i][j+1]] for j in range(len(results[i])-1)]
        consecutive_results.append(consecutive_result)
    return np.stack(consecutive_results,axis=0)

def evaluate_one(result_root, model_name, example):
  """Compare one example on one model, returning SSIM and PSNR scores."""

  #evaluate on target image
  example_dir = os.path.join(result_root, model_name, example)
  tgt_file = tf.gfile.Glob(example_dir + '/tgt_image_*')[0]

  tgt_image = tf.convert_to_tensor(load_image(tgt_file), dtype=tf.float32)
  pred_file = tf.gfile.Glob(example_dir + '/output_tgt_*')[0]
  pred_image = tf.convert_to_tensor(load_image(pred_file), dtype=tf.float32)

  metric = elpips.Metric(elpips.elpips_vgg(batch_size=1),back_prop=False)

  ssim = tf.image.ssim(pred_image, tgt_image, max_val=255.0)
  psnr = tf.image.psnr(pred_image, tgt_image, max_val=255.0)
  elpips_score = metric.forward(tf.expand_dims(pred_image,0), tf.expand_dims(tgt_image,0))

  with tf.Session() as sess:
      return sess.run(ssim).item(), sess.run(psnr).item(), sess.run(elpips_score).item()

def evaluate_consecutive_one(result_root, model_name, example):
  """Compare one pair of consecutive frames on one model, returning the difference of their blurred depth and rgb images."""

  # evaluate on target image
  frame_dirs = [os.path.join(result_root, model_name, example[i]) for i in range(len(example))]
  depth_frame1 = tf.gfile.Glob(frame_dirs[0] + '/output_depth_*')
  depth_frame2 = tf.gfile.Glob(frame_dirs[1] + '/output_depth_*')
  tgt_frame1 = tf.gfile.Glob(frame_dirs[0] + '/output_tgt_*')
  tgt_frame2 = tf.gfile.Glob(frame_dirs[1] + '/output_tgt_*')
  blurred_tgt1_idx = 1 if 'blurred' in tgt_frame1[1] else 0
  blurred_depth1_idx = 1 if 'blurred' in depth_frame1[1] else 0
  blurred_tgt2_idx = 1 if 'blurred' in tgt_frame2[1] else 0
  blurred_depth2_idx = 1 if 'blurred' in depth_frame2[1] else 0

  tgt_blurred_f1 = tf.convert_to_tensor(load_image(tgt_frame1[blurred_tgt1_idx]), dtype=tf.float32)
  depth_blurred_f1 = tf.convert_to_tensor(load_image(depth_frame1[blurred_depth1_idx]), dtype=tf.float32)
  tgt_blurred_f2 = tf.convert_to_tensor(load_image(tgt_frame2[blurred_tgt2_idx]), dtype=tf.float32)
  depth_blurred_f2 = tf.convert_to_tensor(load_image(depth_frame2[blurred_depth2_idx]), dtype=tf.float32)

  tgt_diff = tf.abs(tgt_blurred_f1 - tgt_blurred_f2)
  depth_diff = tf.abs(depth_blurred_f1 - depth_blurred_f2)

  height, width, channels = tgt_diff.get_shape().as_list()
  tgt_diff = tf.reduce_sum(tgt_diff)/ (height * width * channels)
  depth_diff = tf.reduce_sum(depth_diff)/ (height * width * channels)

  with tf.Session() as sess:
      return sess.run(tgt_diff).item(), sess.run(depth_diff).item()

def evaluate_consecutive_example_pair(result_root, model_names, example):
    """example = [frame1-file-name,frame2-file-name]"""
    tf.reset_default_graph()
    tgt_diffs = []
    depth_diffs = []

    tf.logging.info('Starting with %s and %s', example[0], example[1])
    for model_name in model_names:
        tgt_diff, depth_diff = evaluate_consecutive_one(FLAGS.result_root, model_name, example)
        tgt_diffs.append([example[0], tgt_diff]) #[the first frame name, the difference score between it and its next frame]
        depth_diffs.append([example[0], depth_diff])

    return tgt_diffs, depth_diffs

def evaluate_example(result_root, model_names, example):

  tf.reset_default_graph()

  ssims = []
  psnrs = []
  elpipss = []

  tf.logging.info('Starting %s', example)
  for model_name in model_names:
      ssim, psnr, elpips_score = evaluate_one(result_root, model_name, example)
      ssims.append(ssim)
      psnrs.append(psnr)
      elpipss.append(elpips_score)

  return ssims, psnrs, elpipss

def write_output(data):

    with open(FLAGS.output_table, 'w') as f:
        json.dump(data, f)
    tf.logging.info('Output written to %s' % FLAGS.output_table)

def main(_):

  tf.logging.set_verbosity(tf.logging.INFO)
  result_root = FLAGS.result_root
  model_names = FLAGS.model_name.split(',')
  scene_names = FLAGS.videos.split(" ")

  if FLAGS.eval_type == 'on_video':
      '''
      compute the frame-2-frame blurred differences as a metric for temporal consistency across video frames.
      '''
      examples = collect_video_examples(result_root, model_names, scene_names)
      examples = np.sort(np.asarray(examples),axis=1).tolist() #sort the frames temporally by their name
      consecutive_examples = collect_video_consecutive_examples(result_root, model_names, scene_names)

      tf.logging.info('Models: %s', model_names)
      tf.logging.info('%d videos', len(examples))
      tf.logging.info('%d frames', len(examples[0]))

      all_data = []
      videos_data = {}

      num_proc = 20
      pool = Pool(processes=num_proc)
      for i in range(len(scene_names)):
          scene_data = pool.map(
              lambda e: evaluate_consecutive_example_pair(result_root, model_names, e),
              consecutive_examples[i])
          all_data.append(scene_data)
      pool.close()

      for i in range(len(scene_names)):
          depth_diffs = [depth_diff[0][1] for (tgt_diff, depth_diff) in all_data[i]]
          tgt_diffs = [tgt_diff[0][1] for (tgt_diff, depth_diff) in all_data[i]]
          avg_depth_diff = sum(depth_diffs) / len(depth_diffs)
          avg_tgt_diff = sum(tgt_diffs) / len(tgt_diffs)

          sd_depth_diff = math.sqrt(np.var(depth_diffs))
          sd_tgt_diff = math.sqrt(np.var(tgt_diffs))
          data = {
              'avg_depth_diff': avg_depth_diff,
              'avg_tgt_diff': avg_tgt_diff,
              'sd_depth_diff': sd_depth_diff,
              'sd_tgt_diff': sd_tgt_diff
          }
          videos_data[scene_names[i]] = data

      write_output(videos_data)

  else:
      '''
      compute the average ssim, psnr & elpips scores and their variance between test set target-view re-renderings and ground truth output.
      '''

      examples = collect_examples(result_root, model_names)
      examples.sort()

      tf.logging.info('Models: %s', model_names)
      tf.logging.info('%d examples', len(examples))

      num_proc = 20
      pool = Pool(processes=num_proc)
      all_data = pool.map(lambda e: evaluate_example(result_root, model_names, e), examples)
      pool.close()

      ssims = [ssim[0] for (ssim, psnr, elpips_score) in all_data]
      psnrs = [psnr[0] for (ssim, psnr, elpips_score) in all_data]
      elpipss = [elpips_score[0] for (ssim, psnr, elpips_score) in all_data]

      avg_ssim = sum(ssims)/len(ssims)
      avg_psnr = sum(psnrs)/len(psnrs)
      avg_elpips = sum(elpipss)/len(elpipss)

      var_ssim = np.var(ssims)
      var_psnr = np.var(psnrs)
      var_elpips = np.var(elpipss)

      data = {
          'model_names': model_names,
          'avg ssim': avg_ssim,
          'avg psnr': avg_psnr,
          'avg elpips': avg_elpips,
          'var ssim': var_ssim,
          'var psnr': var_psnr,
          'var elpips': var_elpips
      }

      write_output(data)

if __name__ == '__main__':
  app.run()
