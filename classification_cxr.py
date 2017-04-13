import os
import numpy as np
import tensorflow as tf
from Model import ChinaCXRDataset
from sklearn.utils import shuffle

image_width = 640
image_height = 480
pixel_depth = 255.0  # Number of levels per pixel.

cxr_model = ChinaCXRDataset("CXR_png")

if os.path.isfile("CXR_png_gray.pickle"):
    cxr_model.load_from_pickle("CXR_png_gray.pickle")
else:
    cxr_model.load_images(image_width, image_height, pixel_depth, convert_to_gray=True)
    cxr_model.separate_test_dataset(200)
    cxr_model.save(dataset_filename="CXR_png_gray.pickle")

train_dataset, train_labels = cxr_model.random_images(120)
valid_dataset, valid_labels = cxr_model.random_images(120)
test_dataset, test_labels = cxr_model.random_images(120, test_images=True)
num_labels = 2


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_width, image_height, 1)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

batch_size = 16
kernel_size = 5
depth = 16
num_hidden = 64
num_channels = 1

graph1 = tf.Graph()
with graph1.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_width, image_height, num_channels),
                                      name="input")
    tf_train_labels = tf.placeholder(tf.float32,
                                     shape=(batch_size, num_labels),
                                     name="labels")
    tf.summary.image('train_input', tf_train_dataset, 3)
    global_step = tf.Variable(0, trainable=False)
    train = tf.placeholder(tf.bool)
    starter_learning_rate = 0.5
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               500, 0.96, staircase=True)

    def conv(data, patch_size, num_channels, depth, name="conv"):
        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[depth]), name="B")
            wx = tf.nn.conv2d(data, w, [1, 2, 2, 1], padding='SAME')  # zero padded to keep ratio same
            activation = tf.nn.relu(wx + b)
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activation", activation)
            return activation

    def fc_layer(data, width, height, name="fc"):
        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal([width, height], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[height]), name="B")
            mul = tf.matmul(data, w)
            activation = tf.nn.relu(mul + b)
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activation", activation)
            return activation

    # Accuracy
    def accuracy(predictions, labels):
        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
            acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", acc)
            return acc

    def model(data):
        conv1 = conv(data, kernel_size, num_channels, depth, name="conv1")
        conv2 = conv(conv1, kernel_size, depth, depth, name="conv2")
        shape = conv2.get_shape().as_list()
        reshape = tf.reshape(conv2, [shape[0], shape[1] * shape[2] * shape[3]], name="reshape_fc")
        reshape = tf.cond(train, lambda: tf.nn.dropout(reshape, keep_prob=0.7), lambda: reshape)
        fc1 = fc_layer(reshape, image_width // 4 * image_height // 4 * depth, num_hidden, name="fc1")
        fc2 = tf.cond(train, lambda: tf.nn.dropout(fc1, keep_prob=0.7), lambda: fc1)
        fc3 = fc_layer(fc2, num_hidden, num_labels, name="fc2")
        return fc3

    # Training computation.
    logits = model(tf_train_dataset)

    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
        tf.summary.scalar("loss", loss)

    # Optimizer.
    with tf.name_scope("optimiser"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.

    train_accuracy = accuracy(logits, tf_train_labels)
    tf.summary.scalar("train_accuracy", train_accuracy)
    merged_summary = tf.summary.merge_all()  # to get all var summaries in one place.


def get_input_in_batch_size(batch_length, is_train=True):
    if is_train is False:
        batch_data, batch_labels = cxr_model.random_images(batch_length, test_images=True, do_shuffle=True)
    else:
        batch_data, batch_labels = cxr_model.random_images(batch_length, do_shuffle=True)
    return reformat(batch_data, batch_labels)

num_steps = 500


def run_training(graph):
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        writer = tf.summary.FileWriter('/tmp/log_simple_stats/5')
        writer.add_graph(session.graph)
        print('Initialized')

        for step in range(num_steps):
            batch_data, batch_labels = get_input_in_batch_size(batch_size, is_train=True)
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, train: True}
            _ = session.run([optimizer], feed_dict=feed_dict)
            if step % 5 == 0:
                batch_data, batch_labels = get_input_in_batch_size(batch_size, is_train=True)
                feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, train: True}
                valid_acc = session.run([train_accuracy], feed_dict=feed_dict)
                l = session.run([loss], feed_dict=feed_dict)
                va = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(va, step)
                print('Validation loss at step %d: %f' % (step, l[0]))
                print('Validation accuracy: %.1f%%' % (valid_acc[0]*100))
        batch_data, batch_labels = get_input_in_batch_size(batch_size, is_train=False)
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, train: False}
        test_acc = session.run([train_accuracy], feed_dict=feed_dict)
        test_loss = session.run([loss], feed_dict=feed_dict)
        print('Test loss at step %d: %f' % (1000, test_loss[0]))
        print('Test accuracy: %.1f%%' % (test_acc[0] * 100))
        te = session.run(merged_summary, feed_dict=feed_dict)
        writer.add_summary(te, 1000)
        print(test_acc)

run_training(graph1)
