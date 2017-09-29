import json
import os
import random

SAMPLE_NUM = 100
DST_DIR = "./data/dev/train/"


def main():
  with open("data/annotations/captions_train2014.json") as f:
    train_annotations = json.load(f)

  sampled_annotations = list(random.sample(train_annotations["annotations"], SAMPLE_NUM))
  sampled_image_ids = list(map(lambda anno: anno["image_id"],
                               sampled_annotations))

  sampled_images = list(filter(lambda record: record["id"] in sampled_image_ids,
                               train_annotations["images"]))

  dev_annotations = {"annotations": sampled_annotations,
                     "images": sampled_images}

  if not os.path.exists(DST_DIR):
    os.mkdir(DST_DIR)

  annotations_dir = os.path.join(DST_DIR, "annotations")
  if not os.path.exists(annotations_dir):
    os.mkdir(annotations_dir)

  with open(os.path.join(annotations_dir, "captions_train2014.json"), "w") as f:
    json.dump(dev_annotations, f)

  for record in sampled_images:
    print(record['file_name'])


if __name__ == '__main__':
  main()
