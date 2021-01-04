# forest-detectecting
[] Object detection drone service for human search based on deep learning in forest areas

() Introduction

o This service detects and classifies surrounding objects (people, rocks, etc.) and areas (forests, hiking trails, grass, etc.) around accident-risk areas and hiking trails by using forest images captured using drones and AI learning datasets. It is a empirical service that develops a learning model to perform and uses the model to determine and inform the location and characteristics of a person detected on the forest image screen.

![Instance Segmentation Sample](resource/web-service.png)

o The service model is based on the Mask R-CNN, which enables object detection and segmentation simultaneously according to the characteristics of forestland images in which many objects (forests, hiking trails, rocks, etc.) without a predetermined shape are distributed.

o In order to develop the model, the objects appearing in the forestland image were classified into a total of 13 classes and learned. The trained model detects class entities defined in the forestland image, and divides the areas of each detected object to recognize the shape.

o This service model was built to demonstrate a service capable of searching for people in forest areas from images taken by drones. Users upload videos related to forest areas directly from the website(ex : service.forest-detect.com) where the service is provided, and you can see how many people are in each screen of the video, and check information on what area people are located in, and download video for including detection informations.

![Instance Segmentation Sample](resource/service-concept.png)


() Necessity

o When searching for accident-prone areas and people in the forest, there may be cases in which the golden time of search and rescue may be missed as the terrain structure is difficult or the exact location of the requester in the forest is difficult to find.

o When the mountain rescue team is dispatched to the field, the location of the requesting person for help is quickly identified, and life-saving training in preparation for a mountain accident using drone equipment is in progress to determine the topography and search for blind spots, and the effectiveness of drone has been confirmed.

o There is a limitation in monitoring a wide area and various drone shooting screens by humans only relying on the naked eye. So, in the detection of mountain accidents, the search and rescue are efficiently performed with AI detection-based intelligent drone monitoring system. 

o In the future, when operating a drone for a specific purpose such as a search for people in forests by upgrading the learning model and service, the drone equipped with an AI model according to the purpose to autonomously drive for the search (stop when a mountain accident is detected, Alarms, etc.) can be applied.

![Instance Segmentation Sample](resource/in-future.png)
