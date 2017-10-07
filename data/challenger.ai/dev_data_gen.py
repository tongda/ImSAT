import json
import os
import random

SAMPLE_NUM = 100
DST_DIR = "."


def main():
  with open("raw/caption_train_annotations_20170902.json", encoding="utf-8") as f:
    train_annotations = json.load(f)

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


if __name__ == '__main__':
  main()
