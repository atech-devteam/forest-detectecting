import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR) 

# Path to coco trained weights file
MODEL_PATH = os.path.join(ROOT_DIR, "pre_trained_c13.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class DetectConfig(Config):
    
    NAME = "forest"
    IMAGES_PER_GPU = 2
    GPU_COUNT =4
    NUM_CLASSES = 1 + 80  # COCO has 80 classes
    STEPS_PER_EPOCH = 100
    LEARNING_RATE = 0.001
    IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + NUM_CLASSES

############################################################
#  Dataset
############################################################

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
            return super(CocoDataset, self).load_mask(image_id)

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

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.
    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
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
    
############################################################
#  Evaluation : Match Rate, APs
############################################################
def evaluate_dataset(dataset_test, model, config):
    image_ids = dataset_test.image_ids
    APs = []
    
    for i in image_ids:
        sum_match_count = 0
        sum_match_rate = 0
        sum_pred_count = 0
    
        image_id = image_ids[i]
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                        modellib.load_image_gt(dataset_test, config, image_id, use_mini_mask=False)
        
        results = model.detect([image], verbose=0)
        r = results[0]
            
        gt_match, pred_match, overlaps = utils.compute_matches(gt_bbox, gt_class_id, gt_mask,
                        r['rois'], r['class_ids'],r['scores'],r['masks'],
                        iou_threshold=0.5, score_threshold=0.5) 

        count_match = 0
        count_pred = len(pred_match)

        for i in pred_match:
            if i >= 0:
                count_match = count_match + 1
       
        match_rate = count_match/count_pred
        print('Detect count : %d, Match count: %d,  Match rate : %.4f'%(count_pred, count_match, match_rate))
        
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                              r['rois'], r['class_ids'], r['scores'], r['masks'],iou_threshold=0.5)
        APs.append(AP)
                       
    sum_pred_count = sum_pred_count + count_pred
    sum_match_count = sum_match_count + count_match
    sum_match_rate = sum_match_rate + match_rate    
    match_rate_avg = sum_match_rate/len(image_ids)
    
    return sum_match_rate, sum_pred_count, match_rate_avg,APs  

def evaluate_image(gt_bbox, gt_class_id, gt_mask, r):
    
    count_match = 0
    gt_match, pred_match, overlaps = utils.compute_matches(gt_bbox, gt_class_id, gt_mask,
                        r['rois'], r['class_ids'],r['scores'],r['masks'],
                        iou_threshold=0.5, score_threshold=0.5) 
    
    count_gt = len(gt_match)

    for i in gt_match:
        if i >= 0:
            count_match = count_match + 1
   
    match_rate = count_match/count_gt
    
    return count_gt, count_match, match_rate, overlaps