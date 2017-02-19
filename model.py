import csv
import cv2

lines = []
with open('./car_training_data2/driving_log.csv') as csvfile:
#with open('./data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  next(reader)
  for line in reader:
    #print(line)
    lines.append(line)

images = []
measurements = []    
for line in lines:
  #print(line[0])
  #print(line[3])  
  for i in range(3):
	  source_path = line[i] #  center, left , right images are used
	  tokens = source_path.split('/')
	  filename = tokens[-1]
	  local_path = './car_training_data2/IMG/' + filename
	  #local_path = './data/IMG/' + filename
	  image = cv2.imread(local_path)
	  images.append(image)
  correction = 0.2
  measurement = float(line[3])
  measurements.append(measurement) # for center image
  measurements.append(measurement+correction) # for left image
  measurements.append(measurement-correction) # for right image

print(len(images))
print(len(measurements))

augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	flipped_image = cv2.flip(image,1)
	flipped_measurement = measurement * (-1.0)
	augmented_images.append(flipped_image)
	augmented_measurements.append(flipped_measurement)

print("After data augmentation:")
print(len(augmented_images))
print(len(augmented_measurements))

import numpy as np

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

#print(X_train.shape)
#print(y_train.shape)

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# normalize input data into range: (-0.5, 0.5)
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
# crop out undesired area in images such as higer area (the top 70 pixel in height) and lower area (the low 25 pixel in height)
model.add(Cropping2D(cropping=((70,25),(0,0))))
# 5x5 filter size, depth 6
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
# 5x5 filter size, depth 18
model.add(Convolution2D(18,5,5,activation='relu'))
model.add(MaxPooling2D())
# 3x3 filter size, depth 30
model.add(Convolution2D(30,3,3,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(0.5))
model.add(Dense(120))
model.add(Dropout(0.5))
model.add(Dense(84))
model.add(Dropout(0.5))
model.add(Dense(1))
# use Adam optimizer 
model.compile(optimizer='adam', loss='mse')

#from keras.utils.visualize_util import plot
#plot(model, to_file='model.png')

use 20 percent of the data for validation
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)

model.save('model.h5')

