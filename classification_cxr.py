import os 
# import pprint
import numpy as np
# import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import misc
from six.moves import cPickle as pickle
import tensorflow as tf


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


batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64
num_channels = 1

graph1 = tf.Graph()
with graph1.as_default():
	# Input data.
	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_width, image_height, num_channels),name="input")
	tf.summary.image('train_input', tf_train_dataset, 3)
	tf_train_labels  = tf.placeholder(tf.float32, shape=(batch_size, num_labels),name="labels")
	# tf_valid_dataset = tf.constant(valid_dataset)
	# tf_test_dataset  = tf.constant(test_dataset)

	def conv(data, patch_size, num_channels, depth, name="conv"):
		with tf.name_scope(name):
			W = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1),name="W")
			B  = tf.Variable(tf.constant(0.1,shape=[depth]),name="B")
			conv = tf.nn.conv2d(data, W, [1, 2, 2, 1], padding='SAME') # zero padded to keep ratio same
			activation  = tf.nn.relu(conv + B)
			tf.summary.histogram("weights", W)
			tf.summary.histogram("biases", B)
			tf.summary.histogram("activation", activation)
			return activation

	def fc_layer(data, width, height, name="fc"):
		with tf.name_scope(name):
			W = tf.Variable(tf.truncated_normal([width, height], stddev=0.1), name="W")
			B  = tf.Variable(tf.constant(0.1, shape=[height]), name="B")
			mul = tf.matmul(data,W)
			activation  = tf.nn.relu(mul + B)
			tf.summary.histogram("weights", W)
			tf.summary.histogram("biases", B)
			tf.summary.histogram("activation", activation)
			return activation

	# Accuracy
	def accuracy(predictions, labels):
		with tf.name_scope("accuracy"):
		    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
		    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		    tf.summary.scalar("accuracy", accuracy)
		    return accuracy

	def model(data):
		conv1 = conv(data, patch_size, num_channels, depth, name="conv1")
		conv2 = conv(conv1, patch_size, depth, depth, name="conv2")
		shape   = conv2.get_shape().as_list()
		reshape = tf.reshape(conv2, [shape[ 0], shape[1] * shape[2] * shape[3]], name="reshape_fc")
		fc1 = fc_layer(reshape, image_width // 4 * image_height // 4 * depth, num_hidden, name="fc1")
		fc2 = fc_layer(fc1, num_hidden, num_labels, name="fc2")
		return fc2
	# Training computation.
	logits = model(tf_train_dataset)

	with tf.name_scope("loss"):
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
		tf.summary.scalar("loss", loss)

	# Optimizer.
	with tf.name_scope("optimiser"):
		optimizer = tf.train.GradientDescentOptimizer(0.0005).minimize(loss)

	# Predictions for the training, validation, and test data.

	train_accuracy = accuracy(logits, tf_train_labels)
	tf.summary.scalar("train_accuracy", train_accuracy)
	merged_summary = tf.summary.merge_all() # to get all var summaries in one place.

num_steps = 100
def get_input_in_batch_size(step, dataset, labels):
	offset = (step * batch_size) % (labels.shape[0] - batch_size)
	batch_data = dataset[offset:(offset + batch_size), :, :, :]
	batch_labels = labels[offset:(offset + batch_size), :]
	return batch_data, batch_labels	

def run_training(graph):
	with tf.Session(graph=graph) as session:
		tf.initialize_all_variables().run()
		writer = tf.summary.FileWriter('/tmp/log_simple_stats/7')
		writer.add_graph(session.graph)

		print('Initialized')
		for step in range(num_steps):	
			batch_data, batch_labels = get_input_in_batch_size(step, train_dataset, train_labels)
			feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
			_ = session.run([optimizer], feed_dict=feed_dict)
			if (step % 5 == 0):
				l = session.run([loss], feed_dict=feed_dict)
				train_acc = train_accuracy.eval( feed_dict=feed_dict)
				tr = session.run(merged_summary, feed_dict=feed_dict)
				batch_data, batch_labels = get_input_in_batch_size(step, valid_dataset, valid_labels)
				feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
				valid_acc = session.run([train_accuracy], feed_dict=feed_dict)
				va = session.run(merged_summary, feed_dict=feed_dict)
				writer.add_summary(va, step)
				writer.add_summary(tr, step)
				print('Minibatch loss at step %d: %f' % (step, l[0]))
				print('Minibatch accuracy: %.1f%%' % train_acc)
				print( valid_acc)
		batch_data, batch_labels = get_input_in_batch_size(step, test_dataset, test_labels)	
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}	
		test_acc = session.run([train_accuracy], feed_dict=feed_dict)
		te = session.run(merged_summary, feed_dict=feed_dict)	
		writer.add_summary(te, step)
		print(test_acc)

run_training(graph1)

