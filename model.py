import csv
import cv2
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D


#lines = []
samples = []
correction = 0.2
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
#		lines.append(line)
		for i in range(3):
			source_path = line[i]
			filename = source_path.split('/')[-1]
			current_path = './data/IMG/' + filename
			image = cv2.imread(current_path)
			steering = float(line[3])

			if (i == 1):
				measurement = steering
			elif (i == 2):
				measurement = steering + correction
			else:
				measurement = steering - correction
			
			samples.append([image, measurement])


#images = []
#measurements = []
#for line in lines:
#	for i in range(3):
#		source_path = line[i]
#		filename = source_path.split('/')[-1]
#		current_path = './data/IMG/' + filename
#		image = cv2.imread(current_path)
#		images.append(image)
#		measurement = float(line[3])
#		measurements.append(measurement)

#augmented_images, augmented_measurements = [], []
augmented_samples = []
for sample in samples:
#	augmented_images.append(image)
#	augmented_measurements.append(measurement)
#	augmented_images.append(cv2.flip(image, 1))
#	augmented_measurements.append(measurement * -1.0)
	augmented_samples.append([sample[0], sample[1]])
	augmented_samples.append([cv2.flip(sample[0], 1), sample[1] * -1.0])


def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
        
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:

				center_image = batch_sample[0] 
				center_angle = float(batch_sample[1])
				images.append(center_image)
				angles.append(center_angle)
			
				

			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)

#train_samples, validation_samples = train_test_split(np.concatenate(np.array(augmented_images), np.array(augmented_measurements)), test_size=0.2)
train_samples, validation_samples = train_test_split(augmented_samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
#X_train = np.array(augmented_images)
#y_train = np.array(augmented_measurements)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
#model.add(MaxPooling2D())
#model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.fit_generator(train_generator, 
		    samples_per_epoch=len(train_samples),
		    validation_data=validation_generator,
		    nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')
exit()
