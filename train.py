"""Train a mask-rcnn model using tensorflow."""
import os
import sys
import time
import numpy as np
import imgaug
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import argparse

ROOT_DIR = os.path.abspath("./")

from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
from mrcnn.model import log

import forest_detect

_DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
_NUM_CLASSES = 13

parser = argparse.ArgumentParser()

parser.add_argument('--train_epochs', type=int, default=2,
                    help='Number of training epochs: ')

parser.add_argument('--steps_per_epoch', type=int, default=2,
                    help='Number of training epochs: ')                    

parser.add_argument('--end_learning_rate', type=float, default=1e-3,
                    help='End learning rate for the optimizer.')

parser.add_argument('--dataset_dir', type=str, default='datasets',
                    help='Path to the directory containing datasets of images to train.')

parser.add_argument('--logs', required=False,
                        default=_DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')

parser.add_argument('--pre_trained_model', type=str, default='pre_trained_c13.h5',
                    help='Path to the pre-trained model checkpoint.')

parser.add_argument('--images_per_gpu', type=int, default=1,
                    help='Number of images on a gpu: ') 

parser.add_argument('--gpu_count', type=int, default=1,
                    help='Number of gpu to be used: ')                    

class ForestDataset(utils.Dataset):
    def load_dataset(self, dataset_dir, subset, class_ids=None, return_coco=False):
        ann_info = COCO(dataset_dir + "/annotations/instances_" + subset + ".json")
        image_dir = dataset_dir + "/" + subset

        if not class_ids:
            class_ids = sorted(ann_info.getCatIds())
                
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(ann_info.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            image_ids = list(ann_info.imgs.keys())

        for i in class_ids:
            self.add_class("coco", i, ann_info.loadCats(i)[0]["name"])
                                                
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, ann_info.imgs[i]['file_name']),
                width=ann_info.imgs[i]["width"],
                height=ann_info.imgs[i]["height"],
                annotations=ann_info.loadAnns(ann_info.getAnnIds(
                        imgIds=[i], catIds=class_ids, iscrowd=None)))
                        
    
    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(ForestDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                
                if m.max() < 1:
                    continue
                
                if annotation['iscrowd']:
                    class_id *= -1
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(ForestDataset, self).image_reference(image_id)

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m 

class ForestConfig(Config):
    def __init__(self, parser):
      super().__init__()
      self.NAME = "forest"

      # We use a GPU with 12GB memory, which can fit two images.
      # Adjust down if you use a smaller GPU.
      self.IMAGES_PER_GPU = FLAGS.images_per_gpu

      # Uncomment to train on 8 GPUs (default is 1)
      self.GPU_COUNT = FLAGS.gpu_count

      # Number of classes (including background)
      self.NUM_CLASSES = 1 + _NUM_CLASSES  # COCO has 80 classes
      self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES
      
      ## 학습 데이터셋 양에 따라 조정함
      self.STEPS_PER_EPOCH = FLAGS.train_epochs
      self.LEARNING_RATE = FLAGS.end_learning_rate


def main(unused_argv):
    config = ForestConfig(FLAGS)
    args_model = os.path.join(ROOT_DIR, FLAGS.pre_trained_model)
    args_dataset = os.path.join(ROOT_DIR, FLAGS.dataset_dir)
    args_logs = FLAGS.logs

    print("<training configs>")
    print("Model: ", args_model)
    print("Dataset: ", args_dataset)
    print("Logs: ", args_logs)

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=args_logs)
    
    # model.load_weights(args_model, by_name=True)
    dataset_train = ForestDataset()
    dataset_train.load_dataset(dataset_dir = args_dataset, subset= "train", class_ids = None)
    dataset_train.prepare()
    
    dataset_val = ForestDataset()
    dataset_val.load_dataset(dataset_dir = args_dataset, subset= "val", class_ids = None)
    dataset_val.prepare()

    augmentation = imgaug.augmenters.Fliplr(0.5)
    
    print("Training Start!!")
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=FLAGS.steps_per_epoch, layers='all', augmentation=False)
    print("Training End!!")

if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  main([sys.argv[0]] + unparsed)