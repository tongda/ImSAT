import json
import os
import pickle

import tensorflow as tf
from tensorflow.contrib.data import Dataset
from tensorflow.python.ops.lookup_ops import HashTable, KeyValueTensorInitializer

from imsat.model import AttendTell, create_loss
from imsat.vggnet import Vgg19


def main():
  with open("data/annotations/captions_train2014.json") as f:
    annotations = json.load(f)

  id_to_filename = {img['id']: img['file_name'] for img in annotations['images']}

  # todo: to achieve totally end-to-end training, we should
  # call `lower` in tensor-style. But TF do not support
  # `string_lower` directly right now.
  cap_fn_pairs = [(process_caption(ann['caption']), os.path.join("image/train", id_to_filename[ann['image_id']]))
                  for ann in annotations['annotations']]

  captions, filenames = list(zip(*cap_fn_pairs))

  caption_dataset: Dataset = Dataset.from_tensor_slices(list(captions))
  filename_dataset: Dataset = Dataset.from_tensor_slices(list(filenames))

  with open(os.path.join("data", 'word_to_idx.pkl'), 'rb') as f:
    word_to_idx = pickle.load(f)

  table = HashTable(KeyValueTensorInitializer(list(word_to_idx.keys()), list(word_to_idx.values()),
                                              key_dtype=tf.string, value_dtype=tf.int32),
                    0)

  def split_sentence(sentence):
    words = tf.string_split(tf.reshape(sentence, (1,))).values
    words = tf.concat([tf.constant(["<START>"]), words, tf.constant(["<END>"])], axis=0)
    return table.lookup(words)

  index_dataset = caption_dataset.map(split_sentence)

  caption_length_dataset = index_dataset.map(lambda t: tf.size(t))

  def decode_image(filename):
    image = tf.image.decode_jpeg(tf.read_file(filename))
    image = tf.image.resize_images(image, [224, 224])
    image = tf.to_float(image)
    return image

  image_dataset = filename_dataset.map(decode_image)

  iterator = Dataset.zip((image_dataset, caption_dataset, index_dataset, caption_length_dataset)).padded_batch(2, ((224, 224, 3), (), (None,), ())).make_initializable_iterator()
  image_tensor, caption_tensor, index_tensor, length_tensor = iterator.get_next()
  vgg_model_path = "/Users/dtong/code/ai/ImSAT/data/imagenet-vgg-verydeep-19.mat"
  vggnet = Vgg19(vgg_model_path)
  vggnet.build(image_tensor)
  model = AttendTell(word_to_idx=word_to_idx)
  outputs = model.build(features=vggnet.features, captions=index_tensor)
  loss_op = create_loss(outputs, index_tensor, length_tensor)

  optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
  trainables = tf.trainable_variables()
  grads = optimizer.compute_gradients(loss_op, trainables)
  global_step = tf.contrib.framework.get_global_step()
  train_op = optimizer.apply_gradients(grads, global_step=global_step)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  sess.run(table.init)
  sess.run(iterator.initializer)
  for i in range(30):
    loss, _ = sess.run([loss_op, train_op])
    print(loss)

    # batch_size = 4
    #
    # captions_batch = captions[:batch_size]
    # image_idxs_batch = image_idxs[:batch_size]
    # features_batch = features[image_idxs_batch]
    # feed_dict = {features_tensor: features_batch,
    #              captions_tensor: captions_batch}

    # print(sess.run(outputs, feed_dict=feed_dict))


def process_caption(caption):
  caption = caption.replace('.', '').replace(',', '').replace("'", "").replace('"', '')
  caption = caption.replace('&', 'and').replace('(', '').replace(")", "").replace('-', ' ')
  caption = " ".join(caption.split())  # replace multiple spaces
  return caption.lower()


if __name__ == '__main__':
  main()
