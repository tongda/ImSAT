import itertools
import json
import os
import random
import pickle

SAMPLE_NUM = 100
DST_DIR = "."


def main():
  with open("raw/caption_train_annotations_20170902.json", encoding="utf-8") as f:
    train_annotations = json.load(f)

  generate_word_dict(train_annotations)
  sample_records(train_annotations)


def sample_records(train_annotations):
  sampled_annotations = list(random.sample(train_annotations, SAMPLE_NUM))
  if not os.path.exists(DST_DIR):
    os.mkdir(DST_DIR)
  annotations_dir = os.path.join(DST_DIR, "annotations")
  if not os.path.exists(annotations_dir):
    os.mkdir(annotations_dir)
  with open(os.path.join(annotations_dir, "caption_train_annotations_20170902.json"), "w") as f:
    json.dump(sampled_annotations, f)

    # for record in sampled_images:
    #   print(record['file_name'])


def generate_word_dict(train_annotations):
  word_to_idx = dict()
  word_to_idx["<NULL>"] = 0
  word_to_idx["<START>"] = 1
  word_to_idx["<END>"] = 2
  chars = set()
  for ann in train_annotations:
    for cap in ann["caption"]:
      for ch in cap:
        chars.add(ch)
  for k, v in zip(range(3, 3 + len(chars)), chars):
    word_to_idx[v] = k
  print("%d characters saved to the dict." % len(chars))
  print("First 20:")
  print(list(itertools.islice(word_to_idx.items(), 20)))
  with open("word_to_idx.pkl", "wb") as f:
    pickle.dump(word_to_idx, f)


if __name__ == '__main__':
  main()
