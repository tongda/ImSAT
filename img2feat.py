import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.data import Dataset
from tensorflow.contrib.framework import assign_from_checkpoint_fn

from imsat.data import ChallengerAI
from imsat.inception_v4 import inception_v4_base, inception_v4_arg_scope
from imsat.layer import spatial_pyramid_pooling


def main():
  data = ChallengerAI("data/challenger.ai")
  img_dataset, caps_dict = data.get_input_fn(mode="train", is_distort=False)()

  it = Dataset.zip((img_dataset, caps_dict['index'])).batch(1).make_one_shot_iterator()
  img_tensor, cap_tensor = it.get_next()

  bin_size = 14
  arg_scope = inception_v4_arg_scope()
  with slim.arg_scope(arg_scope):
    final_conv_layer, end_points = inception_v4_base(img_tensor)
  feat_tensor = spatial_pyramid_pooling(final_conv_layer, [bin_size], mode='avg')
  feat_tensor = tf.reshape(feat_tensor, shape=(-1, bin_size * bin_size, 1536))

  sess = tf.Session()
  variables_to_restore = slim.get_variables_to_restore(exclude=['global_step'])
  init_fn = assign_from_checkpoint_fn("data/model/inception_v4.ckpt", variables_to_restore)
  init_fn(sess)

  tfrecord_filename = 'data/challenger.ai/tfrecords/feat_14x14x1536_inception_v4.tfrecords'
  writer = tf.python_io.TFRecordWriter(tfrecord_filename)

  for i in range(10):
    f, c = sess.run([feat_tensor, cap_tensor])
    example = tf.train.Example(features=tf.train.Features(
      feature={
        'image_features': tf.train.Feature(bytes_list=tf.train.BytesList(value=[f[0].tostring()])),
        'caption_index': tf.train.Feature(bytes_list=tf.train.BytesList(value=[c[0].tostring()]))
      })
    )
    writer.write(example.SerializeToString())
  writer.close()


if __name__ == '__main__':
  main()
