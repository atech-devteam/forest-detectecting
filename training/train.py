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
sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
from mrcnn.model import log
import coco_mt

import warnings
warnings.filterwarnings(action='ignore')
"""
config.DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
"""
parser = argparse.ArgumentParser()

parser.add_argument('--train_epochs', type=int, default=2,
                    help='Number of training epochs: ')

parser.add_argument('--steps_per_epoch', type=int, default=2,
                    help='Number of training epochs: ')                    

parser.add_argument('--end_learning_rate', type=float, default=1e-3,
                    help='End learning rate for the optimizer.')

parser.add_argument('--dataset_dir', type=str, default='datasets',
                    help='Path to the directory containing the PASCAL VOC data tf record.')

parser.add_argument('--pre_trained_model', type=str, default='logs/coco20201209T1527/mask_rcnn_coco_0001.h5',
                    help='Path to the pre-trained model checkpoint.')

parser.add_argument('--images_per_gpu', type=int, default=4,
                    help='Number of images on a gpu: ') 

parser.add_argument('--gpu_count', type=int, default=2,
                    help='Number of gpu to be used: ')                    

_NUM_CLASSES = 13


def main(unused_argv):
    config = coco_mt.CocoConfig()
    config.DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
    config.IMAGES_PER_GPU = FLAGS.images_per_gpu
    config.GPU_COUNT = FLAGS.gpu_count
    config.NUM_CLASSES = 1 + _NUM_CLASSES
    config.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + config.NUM_CLASSES
    config.STEPS_PER_EPOCH = FLAGS.train_epochs
    config.LEARNING_RATE = FLAGS.end_learning_rate
    args_model = os.path.join(ROOT_DIR, FLAGS.pre_trained_model)
    args_dataset = os.path.join(ROOT_DIR, FLAGS.dataset_dir)
    args_logs = config.DEFAULT_LOGS_DIR

    print("<training configs>")
    print("Model: ", args_model)
    print("Dataset: ", args_dataset)
    print("Logs: ", args_logs)

    warnings.filterwarnings(action='ignore')
    # Create model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=args_logs)
    # Load weights
    model.load_weights(args_model, by_name=True)
    dataset_train = coco_mt.CocoDataset()
    dataset_train.load_coco(dataset_dir = args_dataset, subset= "train", class_ids = None)
    dataset_train.prepare()
    dataset_val = coco_mt.CocoDataset()
    dataset_val.load_coco(dataset_dir = args_dataset, subset= "val", class_ids = None)
    dataset_val.prepare()
    augmentation = imgaug.augmenters.Sequential([ 
                imgaug.augmenters.Fliplr(0.3), imgaug.augmenters.Flipud(0.3), 
                imgaug.augmenters.Affine(rotate=(-45, 45)), imgaug.augmenters.Affine(rotate=(-90, 90)), 
                imgaug.augmenters.color.AddToBrightness((-30, 30), 
                to_colorspace=['YCrCb', 'HSV', 'HLS', 'Lab', 'Luv', 'YUV'], 
                from_colorspace='RGB', seed=None, name=None, 
                random_state='deprecated', deterministic='deprecated')
                # imgaug.augmenters.Affine(scale=(0.5, 1.5))
                ])
    print("Training network all")
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=FLAGS.steps_per_epoch, layers='all', augmentation=augmentation)
    time.sleep(5)
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=FLAGS.steps_per_epoch, layers='4+', augmentation=augmentation)
    time.sleep(5)
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10, epochs=FLAGS.steps_per_epoch, layers='all', augmentation=augmentation)
    print("Training All End!!")

if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  main([sys.argv[0]] + unparsed)