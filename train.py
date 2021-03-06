import argparse
import os

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.data import Dataset
from tensorflow.contrib.framework import assign_from_checkpoint_fn
from tensorflow.contrib.learn import Experiment, RunConfig, ModeKeys
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.training import HParams
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.model_fn import EstimatorSpec

from imsat.data import COCO, ChallengerAI
from imsat.hook import IteratorInitializerHook
from imsat.inception_v4 import inception_v4_base, inception_v4_arg_scope
from imsat.layer import spatial_pyramid_pooling
from imsat.model import AttendTell, create_loss


def model_fn_inner(features, labels, mode, params, config):
  feat_tensor = cap_idx_tensor = cap_len_tensor = None
  scaffold = None
  if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
    cap_lens = labels.map(lambda t: tf.size(t))

    pad_size = ((params.bin_size * params.bin_size, 1536), (None,), ())
    batches = Dataset.zip((features, labels, cap_lens)) \
      .shuffle(buffer_size=200 * params.batch_size) \
      .padded_batch(params.batch_size, pad_size)

    if mode == ModeKeys.TRAIN:
      train_iterator = batches \
        .repeat() \
        .make_initializable_iterator()
      feat_tensor, cap_idx_tensor, cap_len_tensor = \
        train_iterator.get_next()
      tf.add_to_collection("train_initializer", train_iterator.initializer)

    if mode == ModeKeys.EVAL:
      val_iterator = batches \
        .make_initializable_iterator()
      feat_tensor, cap_idx_tensor, cap_len_tensor = \
        val_iterator.get_next()
      tf.add_to_collection("val_initializer", val_iterator.initializer)
      scaffold = tf.train.Scaffold(init_op=val_iterator.initializer)

  if mode == ModeKeys.INFER:
    # for infer, we need to get image id.
    batches = features.padded_batch(params.batch_size, ((), (params.bin_size * params.bin_size, 1536)))
    infer_iterator = batches.make_initializable_iterator()
    image_id, feat_tensor = infer_iterator.get_next()
    tf.add_to_collection("infer_initializer", infer_iterator.initializer)

  loss_op = None
  train_op = None
  predictions = None
  model = AttendTell(vocab_size=params.vocab_size,
                     dim_feature=(params.bin_size * params.bin_size, 1536),
                     selector=params.selector,
                     dropout=params.dropout,
                     ctx2out=params.ctx2out,
                     prev2out=params.prev2out,
                     hard_attention=params.hard_attention,
                     mode=mode)
  if mode != ModeKeys.INFER:
    if params.use_sampler:
      outputs = model.build_train(feat_tensor, cap_idx_tensor,
                                  use_generated_inputs=True)
    else:
      outputs = model.build_train(feat_tensor, cap_idx_tensor,
                                  use_generated_inputs=False)
    loss_op = create_loss(outputs, cap_idx_tensor, cap_len_tensor)
    train_op = _get_train_op(loss_op, params.learning_rate, params.hard_attention)
  else:
    outputs = model.build_infer(feat_tensor)
    predictions = tf.argmax(outputs, axis=-1)

  if mode != ModeKeys.INFER:
    return EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss_op,
      train_op=train_op,
      scaffold=scaffold
    )
  else:
    return EstimatorSpec(
      mode=mode,
      predictions={"image_id": image_id, "predictions": predictions},
      loss=loss_op,
      train_op=train_op,
      scaffold=scaffold
    )


def experiment_fn_inner(run_config, hparams):
  if hparams.dataset == "COCO":
    dataset = COCO("data/coco")
  elif hparams.dataset == "challenger.ai":
    dataset = ChallengerAI("data/challenger.ai")
  else:
    raise Exception("Unknown Dataset Name: '%s'." % hparams.dataset)

  hparams.add_hparam("vocab_size", len(dataset.word_to_idx))

  estimator = Estimator(
    model_fn=model_fn_inner,
    params=hparams,
    config=run_config)

  train_init_hook = IteratorInitializerHook("train")
  val_init_hook = IteratorInitializerHook("val")

  experiment = Experiment(
    estimator=estimator,
    train_input_fn=dataset.get_tfrecords_input_fn("train", hparams.bin_size),
    eval_input_fn=dataset.get_tfrecords_input_fn("val", hparams.bin_size),
    train_steps=hparams.train_steps,
    eval_steps=hparams.eval_steps,
    train_steps_per_iteration=hparams.steps_per_eval,
    eval_hooks=[val_init_hook],
  )
  experiment.extend_train_hooks([train_init_hook])
  return experiment


def model_fn(features, labels, mode, params, config):
  feat_tensor = caption_tensor = cap_idx_tensor = cap_len_tensor = None
  scaffold = None
  bin_size = 8

  if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
    cap_lens = labels["index"].map(lambda t: tf.size(t))

    # todo: cannot utilize GPU to accelerate input pipeline, so train 1 by 1
    # def extract_feats(image):
    #   with tf.device("/gpu:0"):
    #     _, end_points = vgg.vgg_16(tf.expand_dims(image, 0),
    #                                is_training=(mode == ModeKeys.TRAIN),
    #                                spatial_squeeze=False)
    #     final_conv_layer = end_points['vgg_16/conv5/conv5_3']
    #     feats = spatial_pyramid_pooling(final_conv_layer, [bin_size], mode='avg')
    #   return tf.reshape(feats, shape=(bin_size * bin_size, tf.shape(final_conv_layer)[-1]))
    # features = features.map(extract_feats)

    datasets = (features, labels["raw"], labels["index"], cap_lens)
    # todo: 512 is the feature depth, should not hard code here
    # pad_size = ((bin_size * bin_size, 512), (), (None,), ())
    pad_size = ((None, None, 3), (), (None,), ())
    # todo: cannot utilize GPU to accelerate input pipeline, so train 1 by 1
    batches = Dataset.zip(datasets) \
      .shuffle(buffer_size=200 * params.batch_size) \
      .padded_batch(1, pad_size)

    if mode == ModeKeys.TRAIN:
      train_iterator = batches \
        .repeat() \
        .make_initializable_iterator()
      feat_tensor, caption_tensor, cap_idx_tensor, cap_len_tensor = \
        train_iterator.get_next()
      tf.add_to_collection("train_initializer", train_iterator.initializer)

    if mode == ModeKeys.EVAL:
      val_iterator = batches \
        .make_initializable_iterator()
      feat_tensor, caption_tensor, cap_idx_tensor, cap_len_tensor = \
        val_iterator.get_next()
      tf.add_to_collection("val_initializer", val_iterator.initializer)
      scaffold = tf.train.Scaffold(init_op=val_iterator.initializer)

  if mode == ModeKeys.INFER:
    batches = features.batch(params.batch_size)
    infer_iterator = batches.make_initializable_iterator()
    feat_tensor = infer_iterator.get_next()
    tf.add_to_collection("infer_initializer", infer_iterator.initializer)

  feat_tensor = _extract_feats(bin_size, feat_tensor, mode)
  if mode == ModeKeys.TRAIN:
    variables_to_restore = slim.get_variables_to_restore(exclude=['global_step'])
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
                     prev2out=params.prev2out,
                     hard_attention=params.hard_attention,
                     mode=mode)
  if mode != ModeKeys.INFER:
    if params.use_sampler:
      outputs = model.build_train(feat_tensor, cap_idx_tensor,
                                  use_generated_inputs=True)
    else:
      outputs = model.build_train(feat_tensor, cap_idx_tensor,
                                  use_generated_inputs=False)
    loss_op = create_loss(outputs, cap_idx_tensor, cap_len_tensor)
    train_op = _get_train_op(loss_op, params.learning_rate, params.hard_attention)
  else:
    outputs = model.build_infer(feat_tensor)
    predictions = tf.argmax(outputs, axis=-1)

  return EstimatorSpec(
    mode=mode,
    predictions=predictions,
    loss=loss_op,
    train_op=train_op,
    scaffold=scaffold
  )


def _extract_feats(bin_size, image_tensor, mode):
  arg_scope = inception_v4_arg_scope()

  with slim.arg_scope(arg_scope):
    final_conv_layer, end_points = inception_v4_base(image_tensor)

  feat_tensor = spatial_pyramid_pooling(final_conv_layer, [bin_size], mode='avg')
  feat_tensor = tf.reshape(feat_tensor, shape=(-1, bin_size * bin_size, 1536))
  return feat_tensor


def _get_train_op(loss_op, lr, hard_attention=True):
  optimizer = tf.train.AdamOptimizer(learning_rate=lr)
  trainables = tf.trainable_variables()
  filtered_vars = [var for var in trainables if "InceptionV4" not in var.name]
  grads = optimizer.compute_gradients(loss_op, filtered_vars)
  global_step = tf.contrib.framework.get_global_step()
  if hard_attention:
    # todo: this loss is not exactly same with "Show Attend and Tell",
    #   another reference: https://zhuanlan.zhihu.com/p/24879932
    baseline = tf.Variable(0., trainable=False, name="baseline_var")
    ema = tf.train.ExponentialMovingAverage(0.9, name="baseline")
    ema_ap = ema.apply([baseline])
    with tf.control_dependencies([ema_ap]):
      baseline_average = ema.average(baseline)
      grads = [((1 + loss_op - baseline_average) * grad, var) if grad is not None else (grad, var)
               for grad, var in grads]
      baseline_update = tf.assign(baseline, loss_op)
      with tf.control_dependencies([baseline_update]):
        train_op = optimizer.apply_gradients(grads, global_step=global_step)
  else:
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
    train_input_fn=dataset.get_input_fn("train", is_distort=True),
    eval_input_fn=dataset.get_input_fn("val"),
    train_steps=hparams.train_steps,
    eval_steps=hparams.eval_steps,
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
  parser.add_argument("--eval-steps", dest="eval_steps", type=int, default=2,
                      help="Evaluation steps.")
  parser.add_argument("--vgg-model-path", dest="vgg_model_path", type=str,
                      default="data/model/inception_v4.ckpt",
                      help="Path of pre-trained VGG parameters.")
  parser.add_argument("--selector", dest="selector", action="store_true",
                      help="Flag of whether to use selector for context.")
  parser.add_argument("--dropout", dest="dropout", action="store_true",
                      help="Flag of whether to use dropout.")
  parser.add_argument("--ctx2out", dest="ctx2out", action="store_true",
                      help="Flag of whether to add context to output.")
  parser.add_argument("--use-sampler", dest="use_sampler", action="store_true", default=False,
                      help="Flag of whether to add context to output.")
  parser.add_argument("--hard-attention", dest="hard_attention", action="store_true",
                      help="Flag of whether to use hard attention.")
  parser.add_argument("--prev2out", dest="prev2out", action="store_true",
                      help="Flag of whether to add previous state to output.")
  parser.add_argument("--dataset", dest="dataset", type=str, default="COCO",
                      help="Dataset name: COCO or challenger.ai")
  return parser


def get_model_dir(parsed_args):
  exp_name = ("dataset_" + str(parsed_args.dataset) +
              "-hard_attention_" + str(parsed_args.hard_attention) +
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
  session_config.gpu_options.allow_growth = True
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
    dataset=parsed_args.dataset,
    eval_steps=parsed_args.eval_steps,
    hard_attention=parsed_args.hard_attention,
    use_sampler=parsed_args.use_sampler,
    bin_size=14
  )

  learn_runner.run(
    experiment_fn=experiment_fn_inner,
    run_config=run_config,
    schedule="continuous_train_and_eval",
    hparams=params
  )


if __name__ == '__main__':
  main()
