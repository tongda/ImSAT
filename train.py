import argparse
import os

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

from imsat.data import COCO, ChallengerAI
from imsat.hook import IteratorInitializerHook
from imsat.model import AttendTell, create_loss


def model_fn(features, labels, mode, params, config):
  image_tensor = caption_tensor = cap_idx_tensor = cap_len_tensor = None
  scaffold = None

  if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
    cap_lens = labels["index"].map(lambda t: tf.size(t))
    datasets = (features, labels["raw"], labels["index"], cap_lens)
    pad_size = ((224, 224, 3), (), (None,), ())
    batches = Dataset.zip(datasets) \
      .shuffle(buffer_size=10 * params.batch_size) \
      .padded_batch(params.batch_size, pad_size)

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

  if mode == ModeKeys.INFER:
    batches = features.batch(params.batch_size)
    infer_iterator = batches.make_initializable_iterator()
    image_tensor = infer_iterator.get_next()
    tf.add_to_collection("infer_initializer", infer_iterator.initializer)

  _, end_points = vgg.vgg_16(image_tensor)
  image_features = end_points['vgg_16/conv5/conv5_3']
  image_features = tf.reshape(image_features, shape=[-1, 196, 512])

  if mode == ModeKeys.TRAIN:
    variables_to_restore = slim.get_variables_to_restore(exclude=['fc6', 'fc7', 'fc8'])
    init_fn = assign_from_checkpoint_fn(params.vgg_model_path, variables_to_restore)
    # signature of sc
    scaffold = tf.train.Scaffold(init_fn=lambda _, sess: init_fn(sess))

  loss_op = None
  train_op = None
  predictions = None
  model = AttendTell(vocab_size=params.vocab_size,
                     selector=params.selector,
                     dropout=params.dropout,
                     ctx2out=params.ctx2out,
                     prev2out=params.prev2out)
  if mode != ModeKeys.INFER:
    outputs = model.build_train(image_features, cap_idx_tensor)
    loss_op = create_loss(outputs, cap_idx_tensor, cap_len_tensor)
    train_op = _get_train_op(loss_op, params.learning_rate)
  else:
    outputs = model.build_infer(image_features)
    predictions = tf.argmax(outputs, axis=-1)

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
  if hparams.dataset == "COCO":
    dataset = COCO("data/coco")
  elif hparams.dataset == "challenger.ai":
    dataset = ChallengerAI("data/challenger.ai")
  else:
    raise Exception("Unknown Dataset Name: '%s'." % hparams.dataset)

  hparams.add_hparam("vocab_size", len(dataset.word_to_idx))

  estimator = Estimator(
    model_fn=model_fn,
    params=hparams,
    config=run_config)

  train_init_hook = IteratorInitializerHook("train")
  val_init_hook = IteratorInitializerHook("val")

  experiment = Experiment(
    estimator=estimator,
    train_input_fn=dataset.get_input_fn("train"),
    eval_input_fn=dataset.get_input_fn("val"),
    train_steps=hparams.train_steps,
    eval_steps=500,
    train_steps_per_iteration=hparams.steps_per_eval,
    eval_hooks=[val_init_hook],
  )
  experiment.extend_train_hooks([train_init_hook])
  return experiment


def get_parser():
  parser = argparse.ArgumentParser(description="ImSAT trainer.")
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
  parser.add_argument("--selector", dest="selector", action="store_true",
                      help="Flag of whether to use selector for context.")
  parser.add_argument("--dropout", dest="dropout", action="store_true",
                      help="Flag of whether to use dropout.")
  parser.add_argument("--ctx2out", dest="ctx2out", action="store_true",
                      help="Flag of whether to add context to output.")
  parser.add_argument("--prev2out", dest="prev2out", action="store_true",
                      help="Flag of whether to add previous state to output.")
  parser.add_argument("--dataset", dest="dataset", type=str, default="COCO",
                      help="Dataset name: COCO or challenger.ai")
  return parser


def get_model_dir(parsed_args):
  exp_name = ("dataset_" + str(parsed_args.dataset) +
              "-selector_" + str(parsed_args.selector) +
              "-dropout_" + str(parsed_args.dropout) +
              "-ctx2out_" + str(parsed_args.ctx2out) +
              "-prev2out_" + str(parsed_args.prev2out) +
              "-lr_" + str(parsed_args.lr))

  return os.path.join(parsed_args.model_dir, exp_name)


def main():
  tf.logging.set_verbosity(tf.logging.DEBUG)

  parsed_args = get_parser().parse_args()

  session_config = tf.ConfigProto(allow_soft_placement=True)
  run_config = RunConfig(session_config=session_config)
  run_config = run_config.replace(model_dir=get_model_dir(parsed_args))

  params = HParams(
    learning_rate=parsed_args.lr,
    train_steps=parsed_args.train_steps,
    steps_per_eval=parsed_args.steps_per_eval,
    batch_size=parsed_args.batch_size,
    vgg_model_path=parsed_args.vgg_model_path,
    selector=parsed_args.selector,
    dropout=parsed_args.dropout,
    ctx2out=parsed_args.ctx2out,
    prev2out=parsed_args.prev2out,
    dataset=parsed_args.dataset
  )

  learn_runner.run(
    experiment_fn=experiment_fn,
    run_config=run_config,
    schedule="continuous_train_and_eval",
    hparams=params
  )


if __name__ == '__main__':
  main()
