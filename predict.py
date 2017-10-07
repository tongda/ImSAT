import argparse
import json
import os
import pickle

import tensorflow as tf
from tensorflow.contrib.data import Dataset
from tensorflow.contrib.learn import RunConfig
from tensorflow.contrib.training import HParams
from tensorflow.python.estimator.estimator import Estimator

from imsat.hook import IteratorInitializerHook
from train import model_fn


def get_parser():
  parser = argparse.ArgumentParser(description="ImSAT predictor.")
  parser.add_argument("--model-dir", dest="model_dir", type=str,
                      default="ckp-dir/step_10-batch_2-lr_0.001",
                      help="Path of checkpoint.")
  parser.add_argument("--batch-size", dest="batch_size", type=int, default=2,
                      help="Batch size.")
  return parser


def get_input_fn():
  with open("data/annotations/captions_val2014.json") as f:
    annotations = json.load(f)
  id_to_filename = {img['id']: img['file_name'] for img in annotations['images']}
  filenames = [os.path.join("image/val", fn) for fn in id_to_filename.values()]

  def input_fn():
    with tf.variable_scope("input_fn"), tf.device("/cpu:0"):
      filename_dataset = Dataset.from_tensor_slices(list(filenames))

      def decode_image(filename):
        image = tf.image.decode_jpeg(tf.read_file(filename), channels=3)
        image = tf.image.resize_images(image, [224, 224])
        image = tf.to_float(image)
        return image

      image_dataset = filename_dataset.map(decode_image)
    return image_dataset, None

  return id_to_filename.keys(), input_fn


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

  image_ids, input_fn = get_input_fn()
  val_init_hook = IteratorInitializerHook("infer")

  idx_to_word = {v: k for k, v in word_to_idx.items()}
  del word_to_idx

  pred_results = estimator.predict(input_fn, hooks=[val_init_hook])
  all_predicions = []
  num_generated = 0
  for pred in pred_results:
    result = ' '.join([idx_to_word[idx] for idx in pred if idx != 0 and idx != 2])
    all_predicions.append(result)
    num_generated = num_generated + 1
    if num_generated % 1000 == 0:
      print("Generated %d" % num_generated)

  total_results = [{"image_id": img_id, "caption": pred}
                   for img_id, pred
                   in zip(image_ids, all_predicions)]
  with open("result.json", "w") as f:
    json.dump(total_results, f)


if __name__ == '__main__':
  main()
