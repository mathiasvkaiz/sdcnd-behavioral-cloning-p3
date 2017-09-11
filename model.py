import csv
import cv2
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D


lines = []
correction = 0.2
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	print("Process images.")
	for line in reader:
		for i in range(3):
			source_path = line[i]
			filename = source_path.split('/')[-1]
			current_path = './data/IMG/' + filename
			image = cv2.imread(current_path)
			steering = float(line[3])

			# add correction for angles on left and right hand side
			if (i == 0):
				measurement = steering
			elif (i == 1):
				measurement = steering + correction
			else:
				measurement = steering - correction
			
			lines.append([image, measurement])


# Add augmented images for more data
augmented_samples = []
print("Augment images.")
for sample in lines:
	augmented_samples.append([sample[0], sample[1]])
	augmented_samples.append([cv2.flip(sample[0], 1), sample[1] * -1.0])

# generator method for returning only needed data
def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
        
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:

				image = batch_sample[0] 
				angle = float(batch_sample[1])
				images.append(image)
				angles.append(angle)
			
				
			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)

# split data in test and validation
print("Create test and validation set.")
train_samples, validation_samples = train_test_split(augmented_samples, test_size=0.2)

# create generators
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Create model
print("Create model.")
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# compile and fit using generator
print("Start training.")
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, 
		    samples_per_epoch=len(train_samples),
		    validation_data=validation_generator,
		    nb_val_samples=len(validation_samples), nb_epoch=3)

# save model data
print("Save model.")
model.save('model.h5')
exit()
