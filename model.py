import csv
import cv2
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D

# global variables
data_path = './data'
camera = ['center', 'left', 'right']
dict_camera_to_idx = { 'center': 0, 'left': 1, 'right': 2 }
dict_camera_to_correction = { 'center': 0.0, 'left': 0.25, 'right': -0.25 }

print("Process csv data.")
samples = []
with open(data_path + './driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# method for reading images from path
def get_random_image_and_steering(sample):
	# randomly chhose for augmentation
	camera = np.random.choice(camera)

	source_path = sample[dict_camera_to_idx[camera]]
	filename = source_path.split('/')[-1]
	current_path = data_path + '/IMG/' + filename
	
	image = cv2.imread(current_path)
	steering = float(sample[3]) +  dict_camera_to_correction[camera]
	
	return image, steering

# method for randomly flipping image
def decide_to_flip(image, steering):
	flip_prob = np.random.random()
	if flip_prob > 0.5:
	    # flip the image and reverse the steering angle
	    steering = -1 * steering
	    image = cv2.flip(image, 1)
	
	return image, steering

def perform_brightness(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

	random_brightness = .1 + np.random.uniform()
	image[:,:,2] = image[:,:,2] * random_brightness
	image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

	return image

# generator method for returning only needed data
def generator(samples, batch_size=256, img_height=80, img_width=150):
	num_samples = len(samples)
	while (True):
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:

				image, steering = get_random_image_and_steering(batch_sample)
				image, steering = decide_to_flip(image, steering)

				#resize image
				resized_image = cv2.resize(center_image, (img_height, img_width), interpolation=cv2.INTER_AREA)

				# perform more augmentations
				augmented_image = perform_brightness(image)

				# construct arrays
				images.append(augmented_image)
				steerings.append(steering)
			
			# create train arrays
			X_train = np.array(images)
			y_train = np.array(steerings)

			# return train data
			yield sklearn.utils.shuffle(X_train, y_train)

# split data in test and validation
print("Create test and validation set.")
train_samples, validation_samples = train_test_split(augmented_samples, test_size=0.2)

# create generators
print("Generate generators.")
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

# Create model
print("Create model.")
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='elu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='elu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='elu'))
model.add(Convolution2D(64,3,3,activation='elu'))
model.add(Convolution2D(64,3,3,activation='elu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# compile and fit
print("Start training.")
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, 
		    samples_per_epoch=len(train_samples),
		    validation_data=validation_generator,
		    nb_val_samples=len(validation_samples), nb_epoch=5)

# save model data
print("Save model.")
model.save('model.h5')
exit()
