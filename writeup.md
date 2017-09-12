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

[image1]: ./examples/cnn-architecture.png "Model Visualization"
[image2]: ./examples/sea_curve.jpg "Sea Curve"
[image3]: ./examples/bridge.jpg "Bridge"
[image4]: ./examples/3labs.jpg "Three labs"
[image5]: ./examples/recovery1.jpg "Recovery 1"
[image6]: ./examples/recovery2.jpg "Recovery 2"
[image7]: ./examples/recovery3.jpg "Recovery 3"
[image8]: ./examples/recovery4.jpg "Recovery 4"
[image9]: ./examples/recovery5.jpg "Recovery 5"
[image10]: ./examples/augmented1.jpg "Augmented 1"
[image11]: ./examples/augmented2.jpg "Augmented 2"

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

The model.py file contains the code for training and saving the convolution neural network. The file shows the data preparation and pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural based on the Nvidia Network

only the input size is different -> 160, 320, 3 (model.py line 103).
This networks consists of 10 layers, including a normalization layer, 5 convolutional layers, and 4 fully connected layers.
The output has size 1 to predict the steering/angle value. (model.py lines 116)

Strided convolutions in the first three convolutional layers with a 2×2 stride are used. After that  a 5×5 kernel, and a non-strided convolution with a 3×3 kernel size in the final two convolutional layers are used. (model.py lines 107 - 111)

The model includes ELU layers to introduce nonlinearity (model.py lines 107 - 111), and the data is normalized in the model using a Keras lambda layer (code line 105).

Also a cropping layer is used to reduce complexity of images processed (model.py line 106)

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 94, 121). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 120).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was based on experince of professional self driving car builders and relies on the not so complex but powerful structure of the nvidia self driving car cnn model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.
In the first step i haven't used a generator and all data was augmented by using all three cameras and flipping all images as well also adjusting the steering.

Tests on the simulator showed that there are several issues like curve before sea
![alt text][image2]

Also the bridge caused much trouble
![alt text][image3]

After that i switched on generator augmenting inside the generator. Inside the generator a image from left, center oder right was randomly choosed and the steering was adjusted accordingly. After that the image was randomly flipped or not and the image was processed adding brightness. No size adjustment was performed.
Using the generator on the basic test data and performing augmentation inside the generato allowed to have a large number of epochs (30 compared to processing all images at once (despite the risk to run out of memory having all images processed).
The randomly choosing of left, center and right images and fliiping combined with the large number of epochs allowed to have a big data set for training. The chance that every state is trained is very high.

Using the Nvidia architecture resulted in appropriate values on test and validation accuracy seeming not to overfit.

The final step was to run the simulator to see how well the car was driving around track one. Having the generator combined with NVidia architecture lead to good results.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model is already disussed in Part 1 (see above)

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image4]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come back into the center in case of steering to the side. These images show what a recovery looks like starting from ... :

![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image10]
![alt text][image11]


After the collection process, I had round about 10000 data points number of data points. Unfortunately using this data lead to bad results in driving behaviour. The errors on test and validation were very low but driving in autonomous mode was a catostrophy. After recorind several runs i switched to the pregiven data set and then it worked fine.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 15 as evidenced by metrics were not improving anymore. I used an adam optimizer so that manually training the learning rate wasn't necessary.
