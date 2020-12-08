"""
- Description
    Detecting Module 
    this module works with ../config.ini, ../class.json 
- config.ini : configuration
- class.json : classes list with color(RGB - Hex code)
- Created on Oct,2020
- Last Modified on Oct, 2020
- Created by Samuel SS337
"""
import configparser
import os
import sys
import numpy as np 
import skimage.io 
from mrcnn import utils
import mrcnn.model as modellib
from . import coco_mt, visualizer as v
import json
import cv2
from skimage.transform import resize as sk_resize
from skimage import img_as_ubyte

class InferenceConfig(coco_mt.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    def __init__(self, mod_config):
        self.NAME = mod_config['MODEL']['NAME']
        self.GPU_COUNT = int(mod_config['MODEL']['GPU_COUNT'])
        self.IMAGES_PER_GPU = int(mod_config['MODEL']['IMAGES_PER_GPU'])
        self.NUM_CLASSES = 1 + int(mod_config['MODEL']['NUM_CLASSES'])
        super().__init__()

class Detector:
    def __init__(self, rootPath):
        self.mod_config = configparser.ConfigParser()  
        self.ROOT_DIR = rootPath
        self.mod_config.read( os.path.join(self.ROOT_DIR, 'config.ini') )            # the app's configuration 
        self.config = InferenceConfig(self.mod_config) # cocoConfig 
        
        self.MODEL_DIR = os.path.join(self.ROOT_DIR, "model")
        self.COCO_MODEL_PATH = os.path.join(self.MODEL_DIR, self.mod_config['MODEL']['MODEL_FILENAME'])
        self.loadModel()
    
    # call this function after setting the configuration
    def loadModel(self):
        self.model = modellib.MaskRCNN(mode="inference", model_dir=self.COCO_MODEL_PATH, config=self.config)
        self.model.load_weights(self.COCO_MODEL_PATH, by_name=True)
        self.model.keras_model._make_predict_function()
        self.class_names = []
        self.strColors = []
        with open(os.path.join(self.ROOT_DIR,'class.json'),'r') as class_json:
            loadedJson = json.load(class_json)
            for c in loadedJson['classes']:
                self.class_names.append(c['class'])
                self.strColors.append(c['color'])
        self.transColors()

    # to see the configuration of coco
    def showCocoConfig(self):
        print(self.config.display())
    
    def transColors(self):
        self.colors = []
        for color in self.strColors:
            red = int(color[1:3], 16)
            green = int(color[3:5], 16)
            blue = int(color[5:], 16)
            self.colors.append((red/255.0, green/255.0, blue/255.0))
        # print(self.colors)

    """
    detectionFromImgSaveToImg detect from single image file and save to a image file.
    inputFileName and outputFileName must include the path.
    """
    def detectionFromImgSaveToImg(self, inputFileName, outputFileName):
        image = skimage.io.imread(inputFileName)
        # Run detection
        results = self.model.detect([image], verbose=1)
        
        r = results[0]
        v.display_instances(image, r['rois'], r['masks'], r['class_ids'], self.class_names, r['scores'], outputFileName=outputFileName, own_colors=self.colors) 
    
    """
    detect from a video file and save to a video file
    this function is not used in this project.
    """
    def detectionFromVideoSaveToVideo(self, inputFileName, outputFileName, sampling_rate=1, ratio=1):
        v_cap = cv2.VideoCapture(inputFileName)
        v_rate = round(v_cap.get(cv2.CAP_PROP_FPS)) # due to FPS is float
        sampling_rate_4extraction = (v_rate / sampling_rate)
        out_v_rate = 1.0

        v_size = ( int(v_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        new_size = tuple([ int(ratio * elm) for elm in v_size ])
        v_out = cv2.VideoWriter(
            os.path.join(outputFileName), 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            out_v_rate,  
            new_size )
        index = 0
        while True:
            ret, frame = v_cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, dsize=new_size, interpolation=cv2.INTER_AREA) # resizing 
            # frame = sk_resize(frame, new_size)
            results = self.model.detect([frame], verbose=1)
            r = results[0]
            # d_frame = v.display_instances(frame, r['rois'], r['masks'], r['class_ids'], self.class_names, r['scores'], own_colors=self.colors) 
            d_frame = v.display_instances_with_cv2(frame, r['rois'], r['masks'], r['class_ids'], self.class_names, r['scores'], self.colors, int(self.mod_config['MODEL']['FONT_SIZE']))
            v_out.write(cv2.resize(d_frame.astype(np.uint8),new_size))

            v_cap.set(1, index)
            index += sampling_rate_4extraction
            print(f"current index is {index}")

        v_cap.release()
        v_out.release()
    

    """
    
    """
    def detectionFromMem2Mem(self, frame, resize_width=1024):
        v_size = frame.shape
        if v_size[1] != resize_width:
            new_size = ( resize_width, int(v_size[0]*resize_width / v_size[1]))
            frame = cv2.resize(frame, dsize=new_size, interpolation=cv2.INTER_AREA)

        results = self.model.detect([frame], verbose=1)
        r = results[0]
        # print(r['class_ids'])
        return r, self.class_names, v.display_instances_with_cv2(img_as_ubyte(frame), r['rois'], r['masks'], r['class_ids'], 
                                            self.class_names, r['scores'], self.colors, float(self.mod_config['MODEL']['FONT_SIZE']))