from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import glob
import sys
import tensorflow as tf

import i3d

import ipdb,re

# path:
#   export LD_LIBRARY_PATH=/usr/local/cudnn/5.1/lib64:/sequoia/data2/gpunodes_shared_libs/cudnn/5.1/lib64:$LD_LIBRARY_PATH
#
# commands:
# generate list (vidlist contains files: I3D_rgb.npy or I3D_rgb_subXXXX.npy):
#   cd /sequoia/data2/gcheron/DALY/I3D/full/ ; ls */I3D_rgb* > in_rgb_list.txt
#   cd /sequoia/data2/gcheron/DALY/I3D/full/ ; ls */I3D_opf* > in_opf_list.txt
#
# extract features:
# python extract_ucf_rgbflow.py --eval_type rgb --vidlist /sequoia/data2/gcheron/DALY/I3D/full/in_rgb_list.txt
# python extract_ucf_rgbflow.py --eval_type flow --vidlist /sequoia/data2/gcheron/DALY/I3D/full/in_opf_list.txt

DATASET = 'AVA'

if DATASET == 'DALY':
   _H_IMAGE_SIZE = 240 # 360
   _W_IMAGE_SIZE = 320 # 640
   root_dir = '/sequoia/data2/gcheron/DALY/I3D/full'
elif DATASET == 'UCF101':
   _H_IMAGE_SIZE = 240 # 224
   _W_IMAGE_SIZE = 320 # 224
   root_dir = '/sequoia/data2/gcheron/UCF101/I3D/full'
elif DATASET == 'AVA':
   _H_IMAGE_SIZE = 240 # 224
   _W_IMAGE_SIZE = 320 # 224
   root_dir = '/sequoia/data2/dataset/AVA/I3D/full'

_NUM_CLASSES = 400
_MAX_LEN= 1000
_ENDPOINT= 'Mixed_4f' # 'Logits' #'Mixed_5c'

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'data/label_map.txt'

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eval_type','lala', 'rgb or flow')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')
tf.flags.DEFINE_string('vidlist','', 'list of videos')


def main(unused_argv):


  tf.logging.set_verbosity(tf.logging.INFO)
  eval_type = FLAGS.eval_type
  imagenet_pretrained = FLAGS.imagenet_pretrained
  movies = FLAGS.vidlist

  print(eval_type,imagenet_pretrained,movies)

  if eval_type not in ['rgb', 'flow']:
    raise ValueError('Bad `eval_type`, must be one of rgb, flow')

  kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

  with open(movies) as f:
      movie_list = f.readlines()
  movie_list = [re.sub(' .*','',x.strip()) for x in movie_list]

  if eval_type == 'rgb':
    # RGB input has 3 channels.
    rgb_input = tf.placeholder(
        tf.float32,
        shape=(1, None, _H_IMAGE_SIZE, _W_IMAGE_SIZE, 3)) # None allows for any frame number
    with tf.variable_scope('RGB'):
      rgb_model = i3d.InceptionI3d(
          _NUM_CLASSES, spatial_squeeze=True, final_endpoint=_ENDPOINT)
      rgb_features, _ = rgb_model(
          rgb_input, is_training=False, dropout_keep_prob=1.0)
    rgb_variable_map = {}
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == 'RGB':
        rgb_variable_map[variable.name.replace(':0', '')] = variable
    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

  if eval_type == 'flow':
    # Flow input has only 2 channels.
    flow_input = tf.placeholder(
        tf.float32,
        shape=(1, None, _H_IMAGE_SIZE, _W_IMAGE_SIZE, 2))
    with tf.variable_scope('Flow'):
      flow_model = i3d.InceptionI3d(
          _NUM_CLASSES, spatial_squeeze=True, final_endpoint=_ENDPOINT)
      flow_features, _ = flow_model(
          flow_input, is_training=False, dropout_keep_prob=1.0)
    flow_variable_map = {}
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == 'Flow':
        flow_variable_map[variable.name.replace(':0', '')] = variable
    flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)


  if eval_type == 'rgb':
    model_features = rgb_features
  elif eval_type == 'flow':
    model_features = flow_features

  if _ENDPOINT == 'Logits': # if we perform prediction
    model_predictions = tf.nn.softmax(model_features)


  for movie in movie_list:
      #movie=movie_list[427]
      input_path = os.path.join(root_dir, movie)
      rres = re.match('([^/]*).*',movie)
      vidname = rres.group(1)

      with tf.Session() as sess:
        feed_dict = {}
        if eval_type == 'rgb':
          if imagenet_pretrained:
            rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
          else:
            rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb'])
    
        if eval_type in 'flow':
          if imagenet_pretrained:
            flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
          else:
            flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
    
        if eval_type == 'rgb':
           o_str = 'I3D_features_RGB'
        if eval_type == 'flow':
           o_str = 'I3D_features_OPF'

        output_name = os.path.join(root_dir, vidname, o_str)

        # check if from sub
        rres = re.match('([^/]*/I3D_.*_)(sub.*)(.npy)',movie)
        if rres:
           output_name += '_' + rres.group(2)

        output_name += '.npy'
    
        #if not(os.path.isfile(output_name)):
        pstr = input_path
        sample = np.load(input_path)
           
        assert sample.shape[1] <= _MAX_LEN
           #if sample.shape[1] > _MAX_LEN:
           #   sample = flow_sample[:,:_MAX_LEN,:,:,:]   
    
        if eval_type == 'rgb':
           feed_dict[rgb_input] = sample
        if eval_type == 'flow':
           feed_dict[flow_input] = sample
    
    
        if _ENDPOINT == 'Logits':
          out_logits, out_predictions = sess.run(
             [model_features, model_predictions],
             feed_dict=feed_dict)
          out_logits = out_logits[0]
          out_predictions = out_predictions[0]
          sorted_indices = np.argsort(out_predictions)[::-1]
          print('Norm of logits: %f' % np.linalg.norm(out_logits))
          print('\nTop classes and probabilities')
          for index in sorted_indices[:20]:
            print(out_predictions[index], out_logits[index], kinetics_classes[index])
    
        else:
          #pdb.set_trace()
          features = sess.run(model_features,feed_dict=feed_dict)
          np.save(output_name,features)
          print(sample.shape,' ----> ',features.shape)
        print(pstr + '\n' + output_name)


if __name__ == '__main__':
  tf.app.run(main)

