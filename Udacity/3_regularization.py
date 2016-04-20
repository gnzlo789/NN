# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import math

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
batch_size = 128
hidden_size1 = 512
hidden_size2 = 1024
hidden_size3 = 512

graph = tf.Graph()
with graph.as_default():	
	# Input data. For the training data, we use a placeholder that will be fed
	# at run time with a training minibatch.
	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)
	
	# Variables.
	# These are the parameters that we are going to be training. The weight
	# matrix will be initialized using random valued following a (truncated)
	# normal distribution. The biases get initialized to zero.
	weights_1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_size1], stddev=math.sqrt(2./(image_size*image_size))))
	biases_1 = tf.Variable(tf.zeros([hidden_size1]))

	weights_2 = tf.Variable(tf.truncated_normal([hidden_size1, hidden_size2], stddev=math.sqrt(2./(hidden_size1))))
	biases_2 = tf.Variable(tf.zeros([hidden_size2]))

	weights_3 = tf.Variable(tf.truncated_normal([hidden_size2, hidden_size3], stddev=math.sqrt(2./(hidden_size2))))
	biases_3 = tf.Variable(tf.zeros([hidden_size3]))

	weights_4 = tf.Variable(tf.truncated_normal([hidden_size3, num_labels], stddev=math.sqrt(2./(hidden_size3))))
	biases_4 = tf.Variable(tf.zeros([num_labels]))

	# Training computation.
	# We multiply the inputs with the weight matrix, and add biases. We compute
	# the softmax and cross-entropy (it's one operation in TensorFlow, because
	# it's very common, and it can be optimized). We take the average of this
	# cross-entropy across all training examples: that's our loss.

	logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
	act_1 = tf.nn.relu(logits_1)
	dropout_1 = tf.nn.dropout(act_1, 0.6)

	logits_2 = tf.matmul(dropout_1, weights_2) + biases_2
	act_2 = tf.nn.relu(logits_2)
	dropout_2 = tf.nn.dropout(act_2, 0.6)

	logits_3 = tf.matmul(dropout_2, weights_3) + biases_3
	act_3 = tf.nn.relu(logits_3)
	dropout_3 = tf.nn.dropout(act_3, 0.6)

	logits = tf.matmul(dropout_3, weights_4) + biases_4

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

	regularizers = (tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(biases_1) +
					 tf.nn.l2_loss(weights_2) + tf.nn.l2_loss(biases_2) +
					 tf.nn.l2_loss(weights_3) + tf.nn.l2_loss(biases_3) +
					 tf.nn.l2_loss(weights_4) + tf.nn.l2_loss(biases_4))
	loss += 5e-4 * regularizers

	# Optimizer.
	# We are going to find the minimum of this loss using gradient descent.
	global_step = tf.Variable(0)  # count the number of steps taken.
	learning_rate = tf.train.exponential_decay(0.5, global_step, 500, 0.96)

	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
	# optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
	
	# Predictions for the training, validation, and test data.
	# These are not part of training, but merely here so that we can report
	# accuracy figures as we train.
	train_prediction = tf.nn.softmax(logits)

	v_a1 = tf.nn.relu(tf.matmul(tf_valid_dataset, weights_1) + biases_1)
	v_a2 = tf.nn.relu(tf.matmul(v_a1, weights_2) + biases_2)
	v_a3 = tf.nn.relu(tf.matmul(v_a2, weights_3) + biases_3)
	valid_prediction = tf.nn.softmax(tf.matmul(v_a3, weights_4) + biases_4)

	t_a1 = tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1)
	t_a2 = tf.nn.relu(tf.matmul(t_a1, weights_2) + biases_2)
	t_a3 = tf.nn.relu(tf.matmul(t_a2, weights_3) + biases_3)
	test_prediction = tf.nn.softmax(tf.matmul(t_a3, weights_4) + biases_4)

num_steps = 15001

# train_dataset = train_dataset[:256] # Extreme cases
# train_labels = train_labels[:256]

with tf.Session(graph=graph) as session:
	# This is a one-time operation which ensures the parameters get initialized as
	# we described in the graph: random weights for the matrix, zeros for the
	# biases. 
	tf.initialize_all_variables().run()
	print('Initialized')
	for step in range(num_steps):
		# Pick an offset within the training data, which has been randomized.
		# Note: we could use better randomization across epochs.
		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

		# Generate a minibatch.
		batch_data = train_dataset[offset:(offset + batch_size), :]
		batch_labels = train_labels[offset:(offset + batch_size), :]
		
		# Prepare a dictionary telling the session where to feed the minibatch.
		# The key of the dictionary is the placeholder node of the graph to be fed,
		# and the value is the numpy array to feed to it.
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}

		# Run the computations. We tell .run() that we want to run the optimizer,
		# and get the loss value and the training predictions returned as numpy
		# arrays.
		_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

		if (step % 1000 == 0):
			print("Minibatch loss at step %d: %f" % (step, l))
			print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
			# Calling .eval() on valid_prediction is basically like calling run(), but
			# just to get that one numpy array. Note that it recomputes all its graph
			# dependencies.
			print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))

	print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))