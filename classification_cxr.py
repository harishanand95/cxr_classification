import os 
# import pprint
import numpy as np
# import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import misc
from six.moves import cPickle as pickle
import tensorflow as tf
from vgg import vgg_16
import inception_resnet_v2
# dir_path = os.path.dirname(os.path.realpath(__file__))
# s = {}
# files = os.listdir(dir_path + "/ClinicalReadings")
# d = {}
# for i in files:
# 	with open(dir_path+"/ClinicalReadings/%s" % i, "r") as text_file:
# 		lines = text_file.readlines()

# 	if len(lines) > 2 :
# 		d[i] = lines

# 	if lines[1] in s:
# 		s[lines[1]] = s[lines[1]] + 1
# 	else:
# 		s[lines[1]] = 1
# print "=== S ==="
# pprint.pprint(s)
# print "=== D ==="
# pprint.pprint(d)




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
			print num_images
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


def accuracy(predictions, labels):
	# shape   = tf.convert_to_tensor(predictions).get_shape().as_list()
	# prediction = tf.reshape(predictions, [shape[0] * shape[1] * shape[2], shape[3]])
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64
num_channels = 1

graph1 = tf.Graph()
with graph1.as_default():
	# Input data.
	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_width, image_height, num_channels))
	tf_train_labels  = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset  = tf.constant(test_dataset)

	# Variables.
	layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
	layer1_biases  = tf.Variable(tf.zeros([depth]))
	layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
	layer2_biases  = tf.Variable(tf.constant(1.0, shape=[depth]))
	layer3_weights = tf.Variable(tf.truncated_normal([image_width // 4 * image_height // 4 * depth, num_hidden], stddev=0.1))
	layer3_biases  = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
	layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
	layer4_biases  = tf.Variable(tf.constant(1.0, shape=[num_labels]))

	# Model.
	def model(data):
		conv    = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
		hidden  = tf.nn.relu(conv + layer1_biases)
		conv    = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
		hidden  = tf.nn.relu(conv + layer2_biases)
		shape   = hidden.get_shape().as_list()
		reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
		hidden  = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
		return tf.matmul(hidden, layer4_weights) + layer4_biases

	# Training computation.
	logits = model(tf_train_dataset)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

	# Optimizer.
	optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

	# Predictions for the training, validation, and test data.
	train_prediction = tf.nn.softmax(logits)

	valid_prediction = tf.nn.softmax(model(tf_valid_dataset))

	test_prediction = tf.nn.softmax(model(tf_test_dataset))

# VGGNET-16 Implementation

# graph2 = tf.Graph();
# with graph2.as_default():
# 	# Input data.
# 	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_width, image_height, num_channels))
# 	tf_train_labels  = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
# 	tf_valid_dataset = tf.constant(valid_dataset)
# 	tf_test_dataset  = tf.constant(test_dataset)

# 	logits ,_ = vgg_16(tf_train_dataset, num_classes=2, spatial_squeeze=False)
# 	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
# 	optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
# 	train_prediction = tf.nn.softmax(logits)

# 	with tf.variable_scope("g2valid") as scope:
# 		_1, _ = vgg_16(tf_valid_dataset, num_classes=2, is_training=False, spatial_squeeze=False)
# 	valid_prediction = tf.nn.softmax(_1)

# 	with tf.variable_scope("g2test") as scope:
# 		_2, _ = vgg_16(tf_test_dataset, num_classes=2, is_training=False, spatial_squeeze=False)
# 	test_prediction  = tf.nn.softmax(_2)

num_steps = 20
def run_training(graph):
	with tf.Session(graph=graph) as session:
		tf.initialize_all_variables().run()
		# summary_writer = tf.summary.FileWriter('log_simple_stats', session.graph)
		print('Initialized')
		for step in range(num_steps):
			offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
			batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
			batch_labels = train_labels[offset:(offset + batch_size), :]
			feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
			_, l, predictions = session.run(
				[optimizer, loss, train_prediction], feed_dict=feed_dict)
			print "Prediction" 
			print predictions
			print "Values" 
			print batch_labels
			# summary_writer.add_summary(session.run(summaries), step)
			if (step % 5 == 0):
				print('Minibatch loss at step %d: %f' % (step, l))
				print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
				print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
		print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

run_training(graph1)

