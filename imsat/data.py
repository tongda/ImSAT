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

        index_dataset = caption_dataset.map(split_sentence)

        def decode_image(filename):
          image = tf.image.decode_jpeg(tf.read_file(filename), channels=3)
          image = tf.image.resize_images(image, [224, 224])
          image = tf.to_float(image)
          return image

        image_dataset = filename_dataset.map(decode_image)
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

  def get_input_fn(self, mode):
    with open(os.path.join(self.data_dir, "annotations/caption_%s_annotations_20170902.json" % mode)) as f:
      annotations = json.load(f)

    img_cap_pairs = []
    for ann in annotations:
      for cap in ann["caption"]:
        img_cap_pairs.append((ann['image_id'], cap))

    random.shuffle(img_cap_pairs)

    img_ids, caps = list(zip(*img_cap_pairs))
    filenames = [os.path.join(self.data_dir, "image/%s" % mode, img_id)
                 for img_id in img_ids]

    def input_fn():
      caption_dataset = Dataset.from_tensor_slices(list(caps))
      filename_dataset = Dataset.from_tensor_slices(filenames)

      def my_split(text):
        text = text.decode("utf-8")
        idx = [self.word_to_idx[ch] for ch in text]
        idx.insert(0, self.word_to_idx['<START>'])
        idx.append(self.word_to_idx['<END>'])
        return np.array(idx, dtype=np.int32)

      # todo: tf has issue with `tf.string_split` with unicode
      #   https://github.com/tensorflow/tensorflow/issues/11399
      #   so I use `py_func` here.
      index_dataset = caption_dataset.map(lambda text: tf.py_func(my_split, [text], tf.int32))

      def decode_image(filename):
        image = tf.image.decode_jpeg(tf.read_file(filename), channels=3)
        image = tf.image.resize_images(image, [224, 224])
        image = tf.to_float(image)
        return image

      image_dataset = filename_dataset.map(decode_image)

      caption_structure = {
        "raw": caption_dataset,
        "index": index_dataset
      }
      return image_dataset, caption_structure

    return input_fn


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