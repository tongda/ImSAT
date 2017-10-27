import tensorflow as tf

from imsat.data import ChallengerAI

bin_size = 14


def plain_test():
  ds = tf.data.TFRecordDataset("data/challenger.ai/tfrecords/train_feat_14x14x1536_inception_v4.tfrecords")

  def parse(exp):
    features = tf.parse_single_example(
      exp,
      # Defaults are not specified since both keys are required.
      features={
        'img_id': tf.FixedLenFeature([], tf.string),
        'raw_img': tf.FixedLenFeature([], tf.string),
        'img_feats': tf.FixedLenFeature([], tf.string),
        'raw_caps': tf.FixedLenFeature([5, ], tf.string),
        'cap_idx': tf.FixedLenFeature([5, ], tf.string),
      })
    img_id = features['img_id']
    img_tensor = tf.reshape(tf.decode_raw(features['img_feats'], tf.float32), [bin_size * bin_size, 1536])
    cap_tensor = tuple(tf.decode_raw(idx, tf.int32) for idx in tf.unstack(features['cap_idx']))
    return img_id, img_tensor, cap_tensor

  ds = ds.map(parse)
  it = ds.make_one_shot_iterator()
  features_tensor = it.get_next()
  sess = tf.Session()
  result = sess.run(features_tensor)
  print(result[0].decode("utf-8"))
  print(result[1].shape)
  print(len(result[2]))


def test():
  dataset = ChallengerAI("data/challenger.ai")
  feats_ds, caps_ds = dataset.get_tfrecords_input_fn("train", bin_size)()

  it = tf.data.Dataset.zip((feats_ds, caps_ds)).make_one_shot_iterator()
  feats_ts, caps_ts = it.get_next()
  sess = tf.Session()
  feats, caps = sess.run([feats_ts, caps_ts])
  print(feats)
  print(caps)


if __name__ == '__main__':
  test()
