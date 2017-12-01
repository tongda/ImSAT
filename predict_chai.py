import argparse
import glob
import json
import os
import pickle

import tensorflow as tf
from tensorflow.contrib.data import Dataset
from tensorflow.contrib.learn import RunConfig
from tensorflow.contrib.training import HParams
from tensorflow.python.estimator.estimator import Estimator

from imsat.data import get_decode_image_fn, ChallengerAI
from imsat.hook import IteratorInitializerHook
from train import model_fn_inner


def get_parser():
  parser = argparse.ArgumentParser(description="ImSAT predictor.")
  parser.add_argument("--model-dir", dest="model_dir", type=str,
                      default="ckp-dir/dataset_challenger.ai-hard_attention_True-selector_True-dropout_True-ctx2out_True-prev2out_True-lr_0.001",
                      help="Path of checkpoint.")
  parser.add_argument("--batch-size", dest="batch_size", type=int, default=2,
                      help="Batch size.")
  parser.add_argument("--selector", dest="selector", action="store_true",
                      help="Flag of whether to use selector for context.")
  parser.add_argument("--dropout", dest="dropout", action="store_true",
                      help="Flag of whether to use dropout.")
  parser.add_argument("--ctx2out", dest="ctx2out", action="store_true",
                      help="Flag of whether to add context to output.")
  parser.add_argument("--prev2out", dest="prev2out", action="store_true",
                      help="Flag of whether to add previous state to output.")
  parser.add_argument("--hard-attention", dest="hard_attention", action="store_true",
                      help="Flag of whether to use hard attention.")
  return parser


def get_input_fn():
  filenames = glob.glob("data/challenger.ai/image/test/*")
  image_ids = [fn.split("/")[-1] for fn in filenames]

  def input_fn():
    with tf.variable_scope("input_fn"), tf.device("/cpu:0"):
      filename_dataset = Dataset.from_tensor_slices(list(filenames))
      image_dataset = filename_dataset.map(get_decode_image_fn(is_training=False))
    return image_dataset, None

  return image_ids, input_fn


def main():
  parsed_args = get_parser().parse_args()
  with open(os.path.join("data/challenger.ai", 'word_to_idx.pkl'), 'rb') as f:
    word_to_idx = pickle.load(f)
  hparams = HParams(vocab_size=len(word_to_idx),
                    batch_size=parsed_args.batch_size,
                    selector=parsed_args.selector,
                    dropout=parsed_args.dropout,
                    ctx2out=parsed_args.ctx2out,
                    prev2out=parsed_args.prev2out,
                    hard_attention=parsed_args.hard_attention,
                    bin_size=14)
  run_config = RunConfig(model_dir=parsed_args.model_dir)
  estimator = Estimator(
    model_fn=model_fn_inner,
    params=hparams,
    config=run_config)

  dataset = ChallengerAI("data/challenger.ai")

  input_fn = dataset.get_tfrecords_test_input_fn(bin_size=hparams.bin_size)
  val_init_hook = IteratorInitializerHook("infer")

  idx_to_word = {v: k for k, v in word_to_idx.items()}
  del word_to_idx

  results = estimator.predict(input_fn, hooks=[val_init_hook])
  all_predicions = []
  image_ids = []
  num_generated = 0
  for batch_result in results:
    image_id, pred = batch_result["image_id"], batch_result["predictions"]
    result = ''.join([idx_to_word[idx] for idx in pred if idx != 0 and idx != 2])
    all_predicions.append(result)
    image_ids.append(image_id.decode("utf-8").split(".")[0])
    num_generated = num_generated + 1
    if num_generated % 1000 == 0:
      print("Generated %d" % num_generated)

  total_results = [{"image_id": img_id, "caption": pred}
                   for img_id, pred
                   in zip(image_ids, all_predicions)]
  with open("result.json", "w", encoding="utf-8") as f:
    json.dump(total_results, f, ensure_ascii=False)


if __name__ == '__main__':
  main()
