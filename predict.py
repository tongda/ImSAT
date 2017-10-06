import argparse

import os
import pickle

from tensorflow.contrib.learn import RunConfig
from tensorflow.contrib.training import HParams
from tensorflow.python.estimator.estimator import Estimator

from imsat.hook import IteratorInitializerHook
from train import model_fn, get_input_fn


def get_parser():
  parser = argparse.ArgumentParser(description="ImSAT predictor.")
  parser.add_argument("--model-dir", dest="model_dir", type=str,
                      default="ckp-dir/step_10-batch_2-lr_0.001",
                      help="Path of checkpoint.")
  parser.add_argument("--batch-size", dest="batch_size", type=int, default=2,
                      help="Batch size.")
  return parser


def main():
  parsed_args = get_parser().parse_args()
  with open(os.path.join("data", 'word_to_idx.pkl'), 'rb') as f:
    word_to_idx = pickle.load(f)
  hparams = HParams(vocab_size=len(word_to_idx),
                    batch_size=parsed_args.batch_size)
  run_config = RunConfig(model_dir=parsed_args.model_dir)
  estimator = Estimator(
    model_fn=model_fn,
    params=hparams,
    config=run_config)

  input_fn = get_input_fn(word_to_idx, "val")
  val_init_hook = IteratorInitializerHook("infer")

  all_predicions = []
  pred_results = estimator.predict(input_fn, hooks=[val_init_hook])
  idx_to_word = {v: k for k, v in word_to_idx.items()}
  for pred in pred_results:
    result = ' '.join([idx_to_word[idx] for idx in pred if idx != 0])
    all_predicions.append(result)

  print(all_predicions)


if __name__ == '__main__':
  main()
