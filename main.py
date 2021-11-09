# Name: Chirag Patel
# CS370-101
# Object Detection from an image
# Library Used: OpenCV, which is a huge open-source library for computer vision, machine learning, and image processing.

    # Requirements:
        # pip install opencv-python 
        # Requires: Python >=3.6
        # import the image that you want for object detection in images folder and write the image file name on line 24 [image = cv2.imread('images/your_image_name_here.png')].

    # Dataset 
    # COCO is a large-scale object detection, segmentation, and captioning dataset. COCO has several features:
            # Object segmentation
            # Recognition in context
            # Superpixel stuff segmentation
            # 330K images (>200K labeled)
            # 1.5 million object instances
            # 80 object categories
            # 91 stuff categories
            # 5 captions per image
            # 250,000 people with keypoints

import cv2 #importing the library

thres = 0.5 # Threshold to detect object

cap = cv2.VideoCapture(0) # To get camera access
# Setting certain parameters on how big our image is.
cap.set(3, 640)
cap.set(4, 480)

classNames = [] #empty array that will store all the dataset information
classFile = 'coco.names'
with open(classFile,'rt') as f: #opening and reading the dataset file
    classNames = f.read().rstrip('\n').split('\n') #Writing the data from coco.names file into our empty array with stripping and splitting the data


# Downloaded the models from the open cv website. Reason I am using this because it can run fast on any GPU and has a good accuracy rate when detecing for image
# Helps with the information of what each object should look like
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt' # The mobilenet-ssd model is a Single-Shot multibox Detection (SSD) network intended to perform object detection.
weightsPath = 'frozen_inference_graph.pb'

# Setup values to get good results. This will give us the id of the object detected and from thw id we can fetch the name of that object from our dataset.
net = cv2.dnn_DetectionModel(weightsPath,configPath) # Built-in cv2 function that reads the network model stored in Darknet model files and setting them up for detector code.
# Default parameters for the model
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success,image = cap.read() # Gives us our image
    # This will send our image to our model and then it will give us the predictions.
    classIds, confs, bbox = net.detect(image,confThreshold=thres)
    print(classIds, bbox)

    if len(classIds)!= 0:
    # This is to set up what the drawn box size and colour is and the font/size/colour of the name tag will be
        for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(image,box,color=(0,0,255),thickness=2) # To create a rectangle around the detected image with color red and thickness of 2
            cv2.putText(image,classNames[classId-1].upper(),(box[0]+10,box[1]+30), # This adds a text around the rectangle which writes the detected name of the object
                    cv2.FONT_HERSHEY_TRIPLEX,1,(0,0,255),2) # Chooses the font and color of the text

    cv2.imshow("Output", image) #To show the output image with the results
    cv2.waitKey(1) # Displays the output window infinitely

