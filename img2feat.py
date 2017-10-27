import json
import os
import pickle
import sys

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.framework import assign_from_checkpoint_fn
from tensorflow.python.framework.errors_impl import OutOfRangeError

from imsat.inception_v4 import inception_v4_base, inception_v4_arg_scope
from imsat.layer import spatial_pyramid_pooling


def main(mode):
  data_dir = "data/challenger.ai"
  bin_size = 14
  with open(os.path.join(data_dir, 'word_to_idx.pkl'), 'rb') as f:
    word_to_idx = pickle.load(f)

  with open(os.path.join(data_dir, "annotations/caption_%s_annotations_20170902.json" % mode)) as f:
    annotations = json.load(f)

  image_ids = [ann['image_id'] for ann in annotations]
  caps = [ann['caption'] for ann in annotations]

  def my_split(text):
    text = text.decode("utf-8")
    # todo: take care of the unknown character.
    idx = [word_to_idx.get(ch, 0) for ch in text]
    idx.insert(0, word_to_idx['<START>'])
    idx.append(word_to_idx['<END>'])
    return np.array(idx, dtype=np.int32)

  def parse(img_id, caps):
    filename = os.path.join(data_dir, "image/%s" % mode) + "/" + img_id
    image = tf.image.decode_jpeg(tf.read_file(filename), channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    splitted_caps = tuple(map(lambda c: tf.py_func(my_split, [c], tf.int32, stateful=False),
                              tf.unstack(caps)))
    return {
      'img_id': img_id,
      'raw_img': image,
      'raw_caps': caps,
      'cap_idx': splitted_caps
    }
    # return img_id, image, caps, splitted_caps

  it = tf.data.Dataset.from_tensor_slices((image_ids, caps)).map(parse).make_one_shot_iterator()
  feat_tensor_dict = it.get_next()

  arg_scope = inception_v4_arg_scope()
  with slim.arg_scope(arg_scope):
    final_conv_layer, end_points = inception_v4_base(tf.expand_dims(feat_tensor_dict['raw_img'], 0))
  feats_tensor = spatial_pyramid_pooling(final_conv_layer, [bin_size], mode='avg')
  feats_tensor = tf.reshape(feats_tensor, shape=(-1, bin_size * bin_size, 1536))

  sess = tf.Session()

  variables_to_restore = slim.get_variables_to_restore(exclude=['global_step'])
  init_fn = assign_from_checkpoint_fn("data/model/inception_v4.ckpt", variables_to_restore)
  init_fn(sess)

  tfrecord_filename = 'data/challenger.ai/tfrecords/%s_feat_14x14x1536_inception_v4.tfrecords' % mode
  writer = tf.python_io.TFRecordWriter(tfrecord_filename)

  i = 0
  while True:
    try:
      feature_dict, feats = sess.run((feat_tensor_dict, feats_tensor))
      example = tf.train.Example(features=tf.train.Features(
        feature={
          'img_id': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[feature_dict['img_id']])),
          'raw_img': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[feature_dict['raw_img'].tostring()])),
          'img_feats': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[feats.tostring()])),
          'raw_caps': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=feature_dict['raw_caps'])),
          'cap_idx': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[idx.tostring() for idx in feature_dict['cap_idx']])),
        })
      )
      writer.write(example.SerializeToString())
      print(i)
      i += 1
    except OutOfRangeError as e:
      print(e)
      break

  writer.close()


if __name__ == '__main__':
  if len(sys.argv) == 1:
    main("train")
  else:
    main(sys.argv[1])
