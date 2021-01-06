# Detection for searching accidents in forest areas (Under constructing)

## Introduction

### Overview

> This service detects and classifies surrounding objects (people, rocks, etc.) and areas (forests, hiking trails, grass, etc.) around accident-risk areas and hiking trails by using forest images captured using drones and AI learning datasets. It is a empirical service that develops a learning model to perform and uses the model to determine and inform the location and characteristics of a person detected on the forest image screen.

<img src="/resource/i_web-service.png" width="100%" height="100%" title="---" alt="web-service"></img><br/>

> In order to develop the model, the objects appearing in the forestland image were classified into a total of 13 classes and learned. The trained model detects class entities defined in the forestland image, and divides the areas of each detected object to recognize the shape.

> This service model was built to demonstrate a service capable of searching for people in forest areas from images taken by drones. Users upload videos related to forest areas directly from the website(ex : service.forest-detect.com) where the service is provided, and you can see how many people are in each screen of the video, and check information on what area people are located in, and download video for including detection informations.

> The service model is based on the Mask R-CNN [https://github.com/matterport/Mask_RCNN], which enables object detection and segmentation simultaneously according to the characteristics of forestland images in which many objects (forests, hiking trails, rocks, etc.) without a predetermined shape are distributed.

<img src="/resource/i_system-structure.png" width="100%" height="100%" title="---" alt="structure"></img><br/>


### Service Goal

> When searching for accident-prone areas and people in the forest, there may be cases in which the golden time of search and rescue may be missed as the terrain structure is difficult or the exact location of the requester in the forest is difficult to find.

> When the mountain rescue team is dispatched to the field, the location of the requesting person for help is quickly identified, and life-saving training in preparation for a mountain accident using drone equipment is in progress to determine the topography and search for blind spots, and the effectiveness of drone has been confirmed.

> There is a limitation in monitoring a wide area and various drone shooting screens by humans only relying on the naked eye. So, in the detection of mountain accidents, the search and rescue are efficiently performed with AI detection-based intelligent drone monitoring system. 

> In the future, when operating a drone for a specific purpose such as a search for people in forests by upgrading the learning model and service, the drone equipped with an AI model according to the purpose to autonomously drive for the search (stop when a mountain accident is detected, Alarms, etc.) can be applied.

<img src="/resource/i_in-future.png" width="100%" height="100%" title="px(픽셀) 크기 설정" alt="in future"></img><br/>


***
## Dataset for the AI service development


### Original dataset
> In the development of this service, the dataset built through the AI data bulid project for autonomous flying drones.

> Raw dataset(including videos recorded by drone) description : https://aihub.or.kr/aidata/8049


### Dataset format
> For AI model training, it is easy to apply the original dataset to the forestry service learning network, and it is used by converting it into the relatively widely known MS coco format.

> The transformation raw dataset to coco format can be performed with the annotation tool provided in the AI data bulid project.

> The dataset converted for forestry service is composed of the following format, and there are some differences in the original data set and detection object list, annotation format, and file composition.


### List of detectable objects (classes)
|<center> Dataset </center>|<center>Class Names<br>(Label list)</center>|
|-----------------------------------|---------------------------------------------|
|<center>Original Dataset</center><br><center>(28 classes)</center>|tree, person, animal, house, apartment, building, school,office, traffic sign, traffic light, streetlamp, telephone pole, <br>banner, milestone, bridge, tower, car_vechicle, bus_vehicle, truck_vehicle, motorcycle, bike_vehicle, <br>lawn,flower_garden, forest, liver, road, pavement, parking_lot, crosswalk, hiking_trail, trail, flower_bed|
|<center>Forestry Dataset</center><br><center>(13 classes)</center>|tree, person, person_ab, people, forest, road, hiking_trail, rock, rocks, lawn, restarea, parking_lot, car|
<br>

### Dataset Folder Tree
|<center> Dataset </center>|<center> Step </center>|<center>Folder Tree</center>|
|----------------------------------|----------------------------------|----------------------------------|
|Original Dataset|1) Raw video to images| VideoFrames<br> > Image-000001.jpg<br> > Image-000002.jpg<br>  > Image-...<br>|
|Original Dataset|2) Annotations for images| VideoFrames<br>  > Image-000001.jpg<br> > Image-000001.json<br>  > Image-000002.jpg<br> > Image-000002.json<br> > Image-...<br>|
| | | |
|Forestry Dataset|1) Split Train/Validation | Train<br> > Image-000001.jpg<br> > Image-000002.jpg<br> > Image-...<br> <br> Validation<br> > Image-000003.jpg<br>  > Image-000007.jpg<br> > Image-...<br>||
|Forestry Dataset|2) Convert to coco format | Annotation<br>> train.json<br>> validation.json<br><br> Images<br>> Train_Images <br>> Validation_Images<br>  >> Image-000003.jpg<br>  >> Image-000007.jpg<br>  >> Image-...<br>||
<br>

### Annotation(.json) format
<img src="/resource/i_ann-format.png" width="80%" height="80%" title="---" alt="Annotation Format"></img><br/>

***
## Demo Service Overview
### About service
> The web service displays the detection results such as where people are located in the forest image captured by the drone on the screen and provides it as a video file.

<img src="/resource/i_web-service.png" width="100%" height="100%" title="---" alt="web-service"></img><br/>

> Users connect to this service through a web browser and directly upload the video file they want to detect, so that they can check the detection result on the web screen, such as how many people are located on each screen, and what area the people are located in, and download it as a video file.

> In order to provide faster detection results, the result is displayed on the screen as quickly as possible with the Low FPS image instead of all frames of the image by a processor, and at the same time, the high FPS video file including the detection, inference results created by another processor and users can download it.

<img src="/resource/i_user-process.png" width="100%" height="100%" title="---" alt="using service"></img><br/>

### web service url 

### Using service
> Video file upload : Select and upload an video file(mp4) to detect the search result of forestry

<img src="/resource/i_webUI1.png" width="100%" height="100%" title="---" alt="webUI1"></img><br/>

>> 1. Select video file with Explorer or Drag&Drop
>> 2. Start Upload
>> 3. Upolad progress, change to display area when the upload finished

> Detection and Display 

<img src="/resource/i_webUI2.png" width="100%" height="100%" title="---" alt="webUI2"></img><br/>

>> 1. Converts the input video to Low FPS and applies the AI model, and when the processing is complete, the play button is activated
>> 2. AI detection is performed for High FPS video files, and when it is completed, the download button is activated
>> 3. Display detection : AI detection results are displayed on the web screen at Low FPS
>> 4. Play button : Results can be played or stopped when Low FPS conversion is complete
>> 5. Progress bar : You can check the playback time and select the screen of the desired frame
>> 6. Session clear : End all tasks and remove information and data

***
## Overview the forestry AI model 
### About model
> This model detects and classifies objects (people, rocks, etc.) and areas(forest, lawn, etc.) around hiking trails and accident-risk areas in forest video captured by drones.
> Forest vidoes rarely contain objects with a certain shape, such as forests, paths, hiking trails, and rocks.And in many cases, areas such as forests, grass, and rocks in the grass overlap, or the boundaries between objects and areas are ambiguous even with the naked eye of a person.

<img src="/resource/i_forestry-objects.png" width="100%" height="100%" title="---" alt=forest objects"></img><br/>
                                                                                                      
> The AI model for searching for people in forest images should be capable of detecting various types of objects such as people, forests, hiking trails, rocks, etc., and at the same time segmenting the area to understand where people are located.

### Feature of the Mask R-CNN
> Mask R-CNN is a model that combines Faster R-CNN and Fully Convolutional Network (FCN). It is possible to obtain the result of additionally predicting Mask in Classification and BBox Regression predicted by Faster R-CNN and detection and image segmentation can be processed in one operation.
> The model is trained in 3 steps, steps 1 to 2 are Bounding box regression and classification operating in Faster R-CNN, and the last 3 steps are binary classification to determine whether each pixel belongs to the category predicted in step 2.

<img src="/resource/i_mrcnn-network.png" width="100%" height="100%" title="---" alt=Mask R-CNN Network Structure"></img><br/>

***
## Installation for training the model
### About environment 
> The AI model for the search of people in forests was developed in Python, and OpenCV was used for image processing, and based on the Mask R-CNN network that can detect objects and segment regions at the same time.

<img src="/resource/i_environment.png" width="100%" height="100%" title="---" alt=Environment for development"></img><br/>
                                                                                                              
> The development and test environment was configured in the Anaconda virtual environment with Python3.7 (python.org) and Anaconda (anaconda.com) installed on Ubuntu OS.
> For model training using gpu, graphics library toolkit CUDA (10.1) and CUDNN are required.
> For Mask R-CNN library and code, refer to “https://www.github.com/matterport/Mask_RCNN”


### Download and Install
> source download : git clone https://github.com/atech-devteam/forest-detectecting
> Dependencies : pip install --upgrade –r requirements.txt
>> numpy              
>> scipy              
>> cython             
>> Matplotlib         
>> h5py               
>> Pillow             
>> scikit-image       
>> opencv-python      
>> imagug             
>> tensorflow-gpu==1.5
>> keras              
>> jupyter            
>> ipython            
>> pandas             
>> pycocotools        

### Files for training and evaluation
> When the source is installed through git clone, it is created in the following file and folder structure.
> It provide datasets for training, testing and evaluation of AI models and pre-trained model with application code.

<img src="/resource/i_files.png" width="100%" height="100%" title="---" alt=Files for demo"></img><br/>
                                                                                           
***
## Run Training and Evaluation
### Training
> Training settings and process : Basic model(pre-trained weight), Training Dataset, Network configuration

<img src="/resource/i_train-process.png" width="100%" height="100%" title="---" alt=Training Process"></img><br/>
 
> Demo code is provided in two types: python file and ipynb that can be executed by jupyter notebook service(web).

> Running Process with train.py - [options]
>> Epochs : '--train_epochs', type=int, default=2, help='Number of training epochs: '
>> Steps per epoch : '--steps_per_epoch', type=int, default=2,help='Number of training epochs: ' 
>> Learning_rate : '--end_learning_rate', type=float, default=1e-3, help='End learning rate for the optimizer.'
>> Dataset path : '--dataset_dir', type=str, default='datasets', help='Path to the directory containing the PASCAL VOC data tf record.'
>> Pre-trained weight : '--pre_trained_model', type=str, default='logs/coco20201209T1527/mask_rcnn_coco_0001.h5', help='Path to the pre-trained model checkpoint.'
>> Images per GPU : '--images_per_gpu', type=int, default=4, help='Number of images on a gpu: '
>> GPU count : '--gpu_count', type=int, default=2, help='Number of gpu to be used: '
>> Example : python train.py --train_epochs=20 --steps_per_epoch=100 --end_learning_rate=0.001 -dataset_dir="./train/dataset" --pre_trained_model="./train/model —images_per_gpu=3 --gpu_count=2

> Configuration example in train.ipynb

<img src="/resource/i_train-config.png" width="100%" height="100%" title="---" alt=Training Config example"></img><br/>


### Evaluation
> Evaluation settings and process

<img src="/resource/i_evaluate-process.png" width="100%" height="100%" title="---" alt=Evaluation process"></img><br/>
                                                                                                          
> Two test codes are provided, Evaluate_image.ipynb and Evaluate_set.ipynb, and can be executed in the jupyter notebook service (web).                                           >> Evaluate_image : Compare object detection results and GT information for one image, visualize on the image, and visualize the evaluation result with a matching table, etc.
>> Evaluate_set : Compares the object detection result and GT information for the test target data set, calculates and provides evaluation results such as detection rate and recognition rate.

> Settings for evaluation
>> config : model name, gpu count, image per gpu, number of classes
>> saving path of training weight : LOG_MODEL_DIR, PRE_MODEL_PATH
>> dastset path : Dataset_DIR, subset
>> class names : tree, person, forest, ...

> Example : Config in evaluate_set.ipynb (same for evaluate_image.ipynb)

<img src="/resource/i_eval-config.png" width="100%" height="100%" title="---" alt=Evaluation Config example"></img><br/>												    
