#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visual.png "Model Visualization"
[image2]: ./examples/center_2017_02_15_21_58_28_965.jpg "center image"
[image3]: ./examples/left_2017_02_15_21_58_28_965.jpg "left image"
[image4]: ./examples/right_2017_02_15_21_58_28_965.jpg "right Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/combine_image.jpg "Normal Image"
[image7]: ./examples/combine_flipped_image.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with three conv layers with 5x5 filter sizes and corresponding depths : 6, 18, 30 (model.py lines 68-75) 

The model includes RELU layers to introduce nonlinearity (code line 69, 72, 75), and the data is normalized in the model using a Keras lambda layer (code line 65). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 79,81,83). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 86).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to find suitable one from existing models and adapting it to current behavioural cloning task .

To take balance between simplicity and performance, we propose a network which inherits the main features of LeNet5 and Nvidia driving network. My first step was to use a convolution neural network model similar to the LeNet 5 and extend it into style of Nvidia model. I thought this model might be appropriate because the output just one regressor

For handling the aggressive left / right turns when drifting to the left / right of the lane, a constant 0.25 (6.25 deg.) is added to left camera image steering angles, and substracted from right camera image steering angles.

All images (left, center, right) are flipped to provide augmented data. The steering angle is multiplied by -1 for all flipped images.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, dropout layers are added in last three dense layers referened from AlexNet.

The final step was to run the simulator to see how well the car was driving around track one. For the entire navigation, the car can smoothly followed the lane without falling off.

####2. Final Model Architecture

The final model architecture (model.py lines 63-84) consisted of three convolution layers, and each one followed by one mac pooling layer. After flattening, three dense (fully-connected) layers are appended and each one is followed by one dropout layer to reduce overfitting. The details in layers and layer sizes are elaborated in following diagram.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving (from center / left / right cameras):

![alt text][image2]
![alt text][image3]
![alt text][image4]



To augment the data sat, I also flipped horizontally. For example, here is an image that has then been flipped:

Original (center/left/right image)
![alt text][image6]

After flipped (center/left/right image)
![alt text][image7]


After the collection process, I had 24008 number of data points (center+left+right images) and the total number for training became 48016 after applying horizontal flipping for data augmentation. I then preprocessed this data by normalizing into (-0.5, 0.5) range.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by validation error keeping decreasing. I used an adam optimizer so that manually training the learning rate wasn't necessary.


The output video for autonomous testing: https://youtu.be/FbaL0MEuwRI