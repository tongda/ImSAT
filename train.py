import argparse
import json
import os
import pickle

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.data import Dataset
from tensorflow.contrib.framework import assign_from_checkpoint_fn
from tensorflow.contrib.learn import Experiment, RunConfig, ModeKeys
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.slim.nets import vgg
from tensorflow.contrib.training import HParams
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.model_fn import EstimatorSpec
from tensorflow.python.ops.lookup_ops import HashTable, KeyValueTensorInitializer

from imsat.hook import IteratorInitializerHook
from imsat.model import AttendTell, create_loss


def model_fn(features, labels, mode, params, config):
  cap_lens = labels["index"].map(lambda t: tf.size(t))
  datasets = (features, labels["raw"], labels["index"], cap_lens)
  pad_size = ((224, 224, 3), (), (None,), ())
  batches = Dataset.zip(datasets) \
    .shuffle(buffer_size=10 * params.batch_size) \
    .padded_batch(params.batch_size, pad_size)

  image_tensor = caption_tensor = cap_idx_tensor = cap_len_tensor = None
  scaffold = None
  if mode == ModeKeys.TRAIN:
    train_iterator = batches \
      .repeat() \
      .make_initializable_iterator()
    image_tensor, caption_tensor, cap_idx_tensor, cap_len_tensor = \
      train_iterator.get_next()
    tf.add_to_collection("train_initializer", train_iterator.initializer)

  if mode == ModeKeys.EVAL:
    val_iterator = batches \
      .make_initializable_iterator()
    image_tensor, caption_tensor, cap_idx_tensor, cap_len_tensor = \
      val_iterator.get_next()
    tf.add_to_collection("val_initializer", val_iterator.initializer)
    scaffold = tf.train.Scaffold(init_op=val_iterator.initializer)

  _, end_points = vgg.vgg_16(image_tensor)
  image_features = end_points['vgg_16/conv5/conv5_3']
  image_features = tf.reshape(image_features, shape=[-1, 196, 512])

  if mode == ModeKeys.TRAIN:
    variables_to_restore = slim.get_variables_to_restore(exclude=['fc6', 'fc7', 'fc8'])
    init_fn = assign_from_checkpoint_fn(params.vgg_model_path, variables_to_restore)
    # signature of sc
    scaffold = tf.train.Scaffold(init_fn=lambda _, sess: init_fn(sess))

  model = AttendTell(vocab_size=params.vocab_size)
  outputs = model.build(image_features, cap_idx_tensor)
  predictions = tf.argmax(outputs, axis=-1)
  loss_op = None
  train_op = None

  if mode != ModeKeys.INFER:
    loss_op = create_loss(outputs, cap_idx_tensor, cap_len_tensor)
    train_op = _get_train_op(loss_op, params.learning_rate)

  return EstimatorSpec(
    mode=mode,
    predictions=predictions,
    loss=loss_op,
    train_op=train_op,
    scaffold=scaffold
    # eval_metric_ops={"Accuracy": tf.metrics.accuracy(labels=labels['answer'], predictions=predictions, name='accuracy')}
  )


def _get_train_op(loss_op, lr):
  optimizer = tf.train.AdamOptimizer(learning_rate=lr)
  trainables = tf.trainable_variables()
  grads = optimizer.compute_gradients(loss_op, trainables)
  global_step = tf.contrib.framework.get_global_step()
  train_op = optimizer.apply_gradients(grads, global_step=global_step)
  return train_op


def experiment_fn(run_config, hparams):
  with open(os.path.join("data", 'word_to_idx.pkl'), 'rb') as f:
    word_to_idx = pickle.load(f)
  hparams.add_hparam("vocab_size", len(word_to_idx))

  def get_input_fn(mode):
    with open("data/annotations/captions_%s2014.json" % mode) as f:
      annotations = json.load(f)
    id_to_filename = {img['id']: img['file_name'] for img in annotations['images']}
    # todo: to achieve totally end-to-end training, we should
    #   call `lower` in tensor-style. But TF do not support
    #   `string_lower` directly right now.
    cap_fn_pairs = [(process_caption(ann['caption']), os.path.join("image/%s" % mode, id_to_filename[ann['image_id']]))
                    for ann in annotations['annotations']]
    captions, filenames = list(zip(*cap_fn_pairs))

    def input_fn():
      with tf.variable_scope("input_fn"), tf.device("/cpu:0"):
        caption_dataset = Dataset.from_tensor_slices(list(captions))
        filename_dataset = Dataset.from_tensor_slices(list(filenames))

        table_init = KeyValueTensorInitializer(list(word_to_idx.keys()),
                                               list(word_to_idx.values()),
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
          image = tf.image.decode_jpeg(tf.read_file(filename))
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

  estimator = Estimator(
    model_fn=model_fn,
    params=hparams,
    config=run_config)

  train_init_hook = IteratorInitializerHook("train")
  val_init_hook = IteratorInitializerHook("val")

  experiment = Experiment(
    estimator=estimator,
    train_input_fn=get_input_fn("train"),
    eval_input_fn=get_input_fn("val"),
    train_steps=hparams.train_steps,
    eval_steps=500,
    train_steps_per_iteration=hparams.steps_per_eval,
    eval_hooks=[val_init_hook],
  )
  experiment.extend_train_hooks([train_init_hook])
  return experiment


def get_parser():
  parser = argparse.ArgumentParser(description="Windbag trainer.")
  parser.add_argument("--train-steps", dest="train_steps", type=int, default=10,
                      help="Number of steps.")
  parser.add_argument("--learning-rate", dest="lr", type=float, default=0.001,
                      help="Define initial learning rate.")
  parser.add_argument("--model-dir", dest="model_dir", type=str, default="./ckp-dir/",
                      help="define where to save model. "
                           "This is the root dir, every run of experiment will have "
                           "its own sub dir with name generated internally.")
  parser.add_argument("--steps-per-eval", dest="steps_per_eval", type=int, default=5,
                      help="Number of steps between each evaluation,"
                           "`None` by default. if this is `None`, "
                           "evaluation only happens once after train.")
  parser.add_argument("--batch-size", dest="batch_size", type=int, default=2,
                      help="Batch size.")
  parser.add_argument("--vgg-model-path", dest="vgg_model_path", type=str,
                      default="data/model/vgg_16.ckpt",
                      help="Path of pre-trained VGG parameters.")
  return parser


def get_model_dir(parsed_args):
  exp_name = ("step_" + str(parsed_args.train_steps) +
              "-batch_" + str(parsed_args.batch_size) +
              "-lr_" + str(parsed_args.lr))

  return os.path.join(parsed_args.model_dir, exp_name)


def main():
  tf.logging.set_verbosity(tf.logging.DEBUG)

  parsed_args = get_parser().parse_args()

  run_config = RunConfig(log_device_placement=True)
  run_config = run_config.replace(model_dir=get_model_dir(parsed_args))

  params = HParams(
    learning_rate=parsed_args.lr,
    train_steps=parsed_args.train_steps,
    steps_per_eval=parsed_args.steps_per_eval,
    batch_size=parsed_args.batch_size,
    vgg_model_path=parsed_args.vgg_model_path
  )

  learn_runner.run(
    experiment_fn=experiment_fn,
    run_config=run_config,
    schedule="train",
    hparams=params
  )


def process_caption(caption):
  caption = caption.replace('.', '').replace(',', '').replace("'", "").replace('"', '')
  caption = caption.replace('&', 'and').replace('(', '').replace(")", "").replace('-', ' ')
  caption = " ".join(caption.split())  # replace multiple spaces
  return caption.lower()


if __name__ == '__main__':
  main()
