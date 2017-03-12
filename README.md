#**Behavioral Cloning**
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./pics/nvidia_model.png "NVIDIA MODEL"
[image2]: ./pics/loss.png "Loss visualization"

## Project files description
####1.
My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md (this file) summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

The NVIDIA model was studyed and used as training model.

![NVIDIA Model][image1]

In the first layer according to [NVIDIA paper](https://arxiv.org/pdf/1604.07316v1.pdf) the image was normalized.

The model continues with three layers with 2x2 strides and a 5x5 kernel, and two convolutional layers with non-strided convolution with 3x3 kernel size.

Following there are 3 fully connected layers giving as output
the steering angle that will drive our car.

The loss was visualized:
![image2]


### Data preprocessing and augmentation
To train the model I've used data recorded from the simulator.
The Udacity data was used for validation.


The recorded data are quite simliar because the first track has many "turn left" pictures by the track path.

To augment the data I used these steps:


1. **Choose random camera image from center, left and right:** The simulator has three camera views namely; center, left and right views. Using the left and right images, we add and subtract 0.25 to the steering angles respectively to make up for the camera offsets.
2. **Translate image and compensate steering angles:** Since the original image size is 160x320 pixels, we randomly translate image to the left or right and compensate for the translation in the steering angles with 0.008 per pixel of translation. We then crop a region of interest of 120x220 pixel from the image.
3. **Randomly flip image:** In other to balance left and right images, we randomly flip images and change sign on the steering angles.
4. **Brightness Augmentation** We simulate different brightness occasions by converting image to HSV channel and randomly scaling the V channel.
