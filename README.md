# Detection for searching accidents in forest areas

## Introduction

### Overview

> This service detects and classifies surrounding objects (people, rocks, etc.) and areas (forests, hiking trails, grass, etc.) around accident-risk areas and hiking trails by using forest images captured using drones and AI learning datasets. It is a empirical service that develops a learning model to perform and uses the model to determine and inform the location and characteristics of a person detected on the forest image screen.

<img src="/resource/i_web-service.png" width="100%" height="100%" title="---" alt="web-service"></img><br/>

> The service model is based on the Mask R-CNN [https://github.com/matterport/Mask_RCNN], which enables object detection and segmentation simultaneously according to the characteristics of forestland images in which many objects (forests, hiking trails, rocks, etc.) without a predetermined shape are distributed.

> In order to develop the model, the objects appearing in the forestland image were classified into a total of 13 classes and learned. The trained model detects class entities defined in the forestland image, and divides the areas of each detected object to recognize the shape.

> This service model was built to demonstrate a service capable of searching for people in forest areas from images taken by drones. Users upload videos related to forest areas directly from the website(ex : service.forest-detect.com) where the service is provided, and you can see how many people are in each screen of the video, and check information on what area people are located in, and download video for including detection informations.


### Service Goal

> When searching for accident-prone areas and people in the forest, there may be cases in which the golden time of search and rescue may be missed as the terrain structure is difficult or the exact location of the requester in the forest is difficult to find.

> When the mountain rescue team is dispatched to the field, the location of the requesting person for help is quickly identified, and life-saving training in preparation for a mountain accident using drone equipment is in progress to determine the topography and search for blind spots, and the effectiveness of drone has been confirmed.

> There is a limitation in monitoring a wide area and various drone shooting screens by humans only relying on the naked eye. So, in the detection of mountain accidents, the search and rescue are efficiently performed with AI detection-based intelligent drone monitoring system. 

> In the future, when operating a drone for a specific purpose such as a search for people in forests by upgrading the learning model and service, the drone equipped with an AI model according to the purpose to autonomously drive for the search (stop when a mountain accident is detected, Alarms, etc.) can be applied.

<img src="/resource/i_in-future.png" width="100%" height="100%" title="px(픽셀) 크기 설정" alt="in future"></img><br/>


***
## Data for development


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


## Demo Web Service
### About service
> The web service displays the detection results such as where people are located in the forest image captured by the drone on the screen and provides it as a video file.
<img src="/resource/i_web-service.png" width="100%" height="100%" title="---" alt="web-service"></img><br/>

### Using service
> Users connect to this service through a web browser and directly upload the video file they want to detect, so that they can check the detection result on the web screen, such as how many people are located on each screen, and what area the people are located in, and download it as a video file.

