from __future__ import division, print_function, absolute_import
import os
import numpy as np

import matplotlib.image as mpimg
from scipy import misc
from six.moves import cPickle as pickle


import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d
from tflearn.layers.estimator import regression

def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
  
# plt.imshow(gray, cmap = plt.get_cmap('gray'))
# plt.show()

image_width = 640
image_height = 480
pixel_depth = 255.0  # Number of levels per pixel.

def load_images(folder, min_num_images):
	"""Load the data for a single letter label."""
	image_files = os.listdir(folder)
	dataset = np.ndarray(shape=(len(image_files), image_width, image_height),
						 dtype=np.float32)
	labels = np.ndarray(shape=(len(image_files)),
						 dtype=np.int32)
	print(folder)
	num_images = 0
	for image in image_files:
		try:
			image_file = os.path.join(folder, image)
			img = mpimg.imread(image_file)
			gray = rgb2gray(img)
			gray_scaled = misc.imresize(gray, (image_width, image_height))
			image_data = (gray_scaled.astype(float) - pixel_depth / 2) / pixel_depth
			if image_data.shape != (image_width, image_height):
				raise Exception('Unexpected image shape: %s' % str(image_data.shape))
			dataset[num_images, :, :] = image_data
			if str(image_file[-5]) == "1":
				labels[num_images] = 1
			else:
				labels[num_images] = 0
			num_images = num_images + 1
			print(num_images)
		except IOError as e:
			print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

	dataset = dataset[0:num_images, :, :]
	labels = labels[0:num_images]
	if num_images < min_num_images:
	  raise Exception('Many fewer images than expected: %d < %d' %
					  (num_images, min_num_images))

	print('Full dataset tensor:', dataset.shape)
	print('Mean:', np.mean(dataset))
	print('Standard deviation:', np.std(dataset))
	return dataset, labels

def get_dataset():
	set_filename = "CXR_png.pickle"

	if not os.path.isfile(set_filename) :
		dataset , labels = load_images("CXR_png",4)
		data = { "dataset" : dataset, 
				 "labels"  : labels }
		try:
			with open(set_filename, 'wb') as f:
				pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
		except Exception as e:
			print('Unable to save data to', set_filename, ':', e)

	with open(set_filename, 'rb') as f:
		data = pickle.load(f)
		dataset = data["dataset"]
		labels = data["labels"]
		# print "===LABELS==="
		# print labels
		## remove 182
		valid_dataset = np.ndarray(shape=(182, image_width, image_height),
							 dtype=np.float32)
		valid_labels = np.ndarray(shape=(182),
							 dtype=np.int32)
		test_dataset = np.ndarray(shape=(180, image_width, image_height),
							 dtype=np.float32)
		test_labels = np.ndarray(shape=(180),
							 dtype=np.int32)
		train_dataset = np.ndarray(shape=(300, image_width, image_height),
							 dtype=np.float32)
		train_labels = np.ndarray(shape=(300),
							 dtype=np.int32)
		## remove 662
		num_valid = 0
		for i in np.random.randint(low=0, high=662, size=182):
			valid_dataset[num_valid,:,:] = dataset[i,:,:] 
			valid_labels[num_valid] = labels[i]
			num_valid = num_valid + 1

		num_test = 0
		for i in np.random.randint(low=0, high=662, size=180):
			test_dataset[num_test,:,:] = dataset[i,:,:] 
			test_labels[num_test] = labels[i]
			num_test = num_test + 1
		num_train = 0
		for i in np.random.randint(low=0, high=662, size=300):
			train_dataset[num_train,:,:] = dataset[i,:,:] 
			train_labels[num_train] = labels[i]
			num_train = num_train + 1
		
		del dataset  # hint to help gc free up memory
		del labels
		del data
		print('Training set', train_dataset.shape, train_labels.shape)
		print('Validation set', valid_dataset.shape, valid_labels.shape)
		print('Test set', test_dataset.shape, test_labels.shape)
	return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels	


train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_dataset()

num_labels = 2
def reformat(dataset, labels):
	dataset = dataset.reshape((-1, image_width, image_height, 1)).astype(np.float32)
	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
	return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)



# Data loading and preprocessing
# import tflearn.datasets.mnist as mnist
# X, Y, testX, testY = mnist.load_data(one_hot=True)
Y = train_labels
testY = test_labels
X = train_dataset.reshape([-1, image_width, image_height, 1])
testX = test_dataset.reshape([-1, image_width, image_height, 1])

batch_size = 16
image_width = 640
image_height = 480
num_labels = 2
filter_size = 5
depth = 16

# Building convolutional network
network = input_data(shape=[None, image_width, image_height, 1], name='input')
network = conv_2d(network, depth, filter_size, [1,2,2,1], activation='relu', regularizer="L2", padding='SAME')
network = conv_2d(network, depth, filter_size, [1,2,2,1], activation='relu', regularizer="L2", padding='SAME')
network = fully_connected(network, 64)
network = dropout(network, 0.8)
network = fully_connected(network, num_labels, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.001,
                     loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=3)
model.fit({'input': X}, {'target': Y}, n_epoch=20,
           validation_set=({'input': testX}, {'target': testY}),
           snapshot_step=100, show_metric=True, run_id='convnet')

