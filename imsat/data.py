import glob
import json
import os
import pickle
import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import Dataset
from tensorflow.python.ops.lookup_ops import KeyValueTensorInitializer, HashTable


class COCO:
  def __init__(self, data_dir):
    self.data_dir = data_dir
    with open(os.path.join(data_dir, 'word_to_idx.pkl'), 'rb') as f:
      self.word_to_idx = pickle.load(f)

  def get_input_fn(self, mode):
    with open(os.path.join(self.data_dir, "annotations/captions_%s2014.json" % mode)) as f:
      annotations = json.load(f)
    id_to_filename = {img['id']: img['file_name'] for img in annotations['images']}

    # todo: to achieve totally end-to-end training, we should
    #   call `lower` in tensor-style. But TF do not support
    #   `string_lower` directly right now.
    def full_path(ann):
      return os.path.join(self.data_dir, "image/%s" % mode, id_to_filename[ann['image_id']])

    cap_fn_pairs = [(process_caption(ann['caption']), full_path(ann))
                    for ann in annotations['annotations']]
    captions, filenames = list(zip(*cap_fn_pairs))

    def input_fn():
      with tf.variable_scope("input_fn"), tf.device("/cpu:0"):
        caption_dataset = Dataset.from_tensor_slices(list(captions))
        filename_dataset = Dataset.from_tensor_slices(list(filenames))

        table_init = KeyValueTensorInitializer(list(self.word_to_idx.keys()),
                                               list(self.word_to_idx.values()),
                                               key_dtype=tf.string,
                                               value_dtype=tf.int32)
        table = HashTable(table_init, default_value=0)

        def split_sentence(sentence):
          words = tf.string_split(tf.reshape(sentence, (1,))).values
          words = tf.concat([tf.constant(["<START>"]), words, tf.constant(["<END>"])],
                            axis=0)
          return table.lookup(words)

        index_dataset = caption_dataset.map(split_sentence, num_threads=8)

        def decode_image(filename):
          image = tf.image.decode_jpeg(tf.read_file(filename), channels=3)
          # image = tf.image.resize_images(image, [224, 224])
          image = tf.to_float(image)
          return image

        image_dataset = filename_dataset.map(decode_image, num_threads=8)
        caption_structure = {
          "raw": caption_dataset,
          "index": index_dataset
        }
      return image_dataset, caption_structure

    return input_fn


class ChallengerAI:
  def __init__(self, data_dir):
    self.data_dir = data_dir
    with open(os.path.join(data_dir, 'word_to_idx.pkl'), 'rb') as f:
      self.word_to_idx = pickle.load(f)

  def get_tfrecords_input_fn(self, mode, bin_size):
    tfrecords_filenames = glob.glob(os.path.join(self.data_dir,
                                                 "tfrecords/%s_feat_14x14x1536_inception_v4-*.tfrecords" % mode))

    def input_fn():
      ds = tf.data.TFRecordDataset(tfrecords_filenames)

      def parse_feats(exp):
        features = tf.parse_single_example(
          exp,
          # Defaults are not specified since both keys are required.
          features={
            'img_id': tf.FixedLenFeature([], tf.string),
            # 'raw_img': tf.FixedLenFeature([], tf.string),
            'img_feats': tf.FixedLenFeature([], tf.string),
            'raw_caps': tf.FixedLenFeature([5, ], tf.string),
            'cap_idx': tf.FixedLenFeature([5, ], tf.string),
          })
        feats_tensor = tf.reshape(tf.decode_raw(features['img_feats'], tf.float32), [bin_size * bin_size, 1536])
        return feats_tensor

      def parse_caps(exp):
        features = tf.parse_single_example(
          exp,
          # Defaults are not specified since both keys are required.
          features={
            'img_id': tf.FixedLenFeature([], tf.string),
            # 'raw_img': tf.FixedLenFeature([], tf.string),
            'img_feats': tf.FixedLenFeature([], tf.string),
            'raw_caps': tf.FixedLenFeature([5, ], tf.string),
            'cap_idx': tf.FixedLenFeature([5, ], tf.string),
          })

        cap_tensor = tf.decode_raw(random.choice(tf.unstack(features['cap_idx'])), tf.int32)

        return cap_tensor

      return ds.map(parse_feats), ds.map(parse_caps)

    return input_fn

  def get_input_fn(self, mode, is_distort=False):
    with open(os.path.join(self.data_dir, "annotations/caption_%s_annotations_20170902.json" % mode)) as f:
      annotations = json.load(f)

    img_cap_pairs = []
    for ann in annotations:
      for cap in ann["caption"]:
        filename = os.path.join(self.data_dir, "image/%s" % mode, ann['image_id'])
        if os.path.exists(filename):
          img_cap_pairs.append((ann['image_id'], cap))
        else:
          print("Image Not Exist: %s" % filename)

    random.shuffle(img_cap_pairs)

    img_ids, caps = list(zip(*img_cap_pairs))
    filenames = [os.path.join(self.data_dir, "image/%s" % mode, img_id)
                 for img_id in img_ids]

    def input_fn():
      caption_dataset = Dataset.from_tensor_slices(list(caps))
      filename_dataset = Dataset.from_tensor_slices(filenames)

      def my_split(text):
        text = text.decode("utf-8")
        # todo: take care of the unknown character.
        idx = [self.word_to_idx.get(ch, 0) for ch in text]
        idx.insert(0, self.word_to_idx['<START>'])
        idx.append(self.word_to_idx['<END>'])
        return np.array(idx, dtype=np.int32)

      # todo: tf has issue with `tf.string_split` with unicode
      #   https://github.com/tensorflow/tensorflow/issues/11399
      #   so I use `py_func` here.
      index_dataset = caption_dataset.map(lambda text: tf.py_func(my_split, [text], tf.int32),
                                          num_threads=8)

      image_dataset = filename_dataset.map(get_decode_image_fn(is_training=is_distort), num_threads=8)

      caption_structure = {
        "raw": caption_dataset,
        "index": index_dataset
      }
      return image_dataset, caption_structure

    return input_fn


def get_decode_image_fn(is_training=True):
  def decode_image(filename):
    image = tf.image.decode_jpeg(tf.read_file(filename), channels=3)
    # image = tf.image.resize_images(image, [224, 224])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if is_training:
      image = distort_image(image)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image

  return decode_image


def distort_image(image):
  """Perform random distortions on an image.

  Args:
    image: A float32 Tensor of shape [height, width, 3] with values in [0, 1).

  Returns:
    distorted_image: A float32 Tensor of shape [height, width, 3] with values in
      [0, 1].
  """
  # Randomly flip horizontally.
  with tf.name_scope("flip_horizontal", values=[image]):
    image = tf.image.random_flip_left_right(image)

  # Randomly distort the colors based on thread id.
  with tf.name_scope("distort_color", values=[image]):
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.032)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)

  return image


def process_caption(caption):
  caption = caption.replace('.', '').replace(',', '').replace("'", "").replace('"', '')
  caption = caption.replace('&', 'and').replace('(', '').replace(")", "").replace('-', ' ')
  caption = " ".join(caption.split())  # replace multiple spaces
  return caption.lower()


if __name__ == '__main__':
  sess = tf.Session()
  chai = ChallengerAI("data/challenger.ai")
  fn = chai.get_input_fn("train")
  features, labels = fn()

  idx_iter = labels["index"].make_initializable_iterator()
  img_iter = features.make_initializable_iterator()

  sess.run(tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS))
  sess.run(idx_iter.initializer)
  sess.run(img_iter.initializer)
  idx_op = idx_iter.get_next()
  img_op = img_iter.get_next()

  for _ in range(10):
    print(sess.run([idx_op, img_op]))
