import random
import pandas
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

import tensorflow as tf
import skflow
import numpy

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

tf.set_random_seed(42)

train = pandas.read_csv('data/titanic_train.csv')
y, X = train['Survived'], train[['Age', 'SibSp', 'Fare']].fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = numpy.array(X_train)
X_test = numpy.array(X_test)
#y_train = numpy.array(y_train).reshape(y_train.shape[-1], 1)
#y_test = numpy.array(y_test).reshape(y_test.shape[-1], 1)

result = []
for elem in y_train:
	if elem == 0:
		result.append([1,0])
	else:
		result.append([0,1])
y_train = numpy.array(result)

result = []
for elem in y_test:
	if elem == 0:
		result.append([1,0])
	else:
		result.append([0,1])
y_test = numpy.array(result)

# Parameters
learning_rate = 0.05
training_epochs = 500
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 10 # 1st layer num features # 256 mnist
n_hidden_2 = 20 # 2nd layer num features # 256 mnist
n_hidden_3 = 10 # 3rd layer num features
n_input = 3 # data input # 784 mnist
n_classes = 2 # 0-1 digits # 10 mnist
dropout = 0.75

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(_X, _weights, _biases):
	layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) #Hidden layer with RELU activation
	#layer_1 = tf.nn.dropout(layer_1, dropout)
	layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])) #Hidden layer with RELU activation
	#layer_2 = tf.nn.dropout(layer_2, dropout)
	layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, _weights['h3']), _biases['b3'])) #Hidden layer with RELU activation
	#layer_3 = tf.nn.dropout(layer_3, dropout)
	return tf.matmul(layer_3, _weights['out']) + _biases['out']


# Store layers weight & bias
weights = {
	'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.1)),
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
	'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=0.1)),
	'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes], stddev=0.1))
}
biases = {
	'b1': tf.Variable(tf.zeros([n_hidden_1])),
	'b2': tf.Variable(tf.zeros([n_hidden_2])),
	'b3': tf.Variable(tf.zeros([n_hidden_3])),
	'out': tf.Variable(tf.zeros([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# Initializing the variables
init = tf.initialize_all_variables()


# Create a summary to monitor cost function
tf.scalar_summary("loss", cost)

# Merge all summaries to a single operator
merged_summary_op = tf.merge_all_summaries()


# Launch the graph
with tf.Session() as sess:
	sess.run(init)

	# Set logs writer into folder /tmp/tensorflow_logs
	summary_writer = tf.train.SummaryWriter('/tmp/tensorflow_logs', graph_def=sess.graph_def)

	# Training cycle
	for epoch in range(training_epochs):
		avg_cost = 0.
		# total_batch = int(mnist.train.num_examples/batch_size)
		total_batch = int(round(float(X_train.shape[0])/batch_size))
		# Loop over all batches
		for i in range(total_batch):
			batch_xs = X_train[i*batch_size:i*batch_size+batch_size]
			batch_ys = y_train[i*batch_size:i*batch_size+batch_size]
			# Fit training using batch data
			sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
			# Compute average loss
			avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
			
			summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
			summary_writer.add_summary(summary_str, epoch*total_batch + i)
		# Display logs per epoch step
		if epoch % display_step == 0:
			print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
	
	print "Optimization Finished!"
	
	# Test model
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	# Calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print "Accuracy:", accuracy.eval({x: X_test, y: y_test})


# Skflow

y, X = train['Survived'], train[['Age', 'SibSp', 'Fare']].fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = numpy.array(X_train)
X_test = numpy.array(X_test)
y_train = numpy.array(y_train).reshape(y_train.shape[-1], 1)
y_test = numpy.array(y_test).reshape(y_test.shape[-1], 1)

classifier = skflow.TensorFlowDNNClassifier(
	hidden_units=[10, 20, 10], 
	n_classes=2, 
	batch_size=128, 
	steps=500,
	optimizer='Adam',
	learning_rate=0.05)
classifier.fit(X_train, y_train)
print(accuracy_score(classifier.predict(X_test), y_test))

# TensorFlowDNNClassifier(batch_size=128, class_weight=None,
#	continue_training=False, early_stopping_rounds=None,
#	hidden_units=[10, 20, 10], keep_checkpoint_every_n_hours=10000,
#	learning_rate=0.05, max_to_keep=5, n_classes=2, num_cores=4,
#	optimizer='SGD', steps=500, tf_master='', tf_random_seed=42,
#	verbose=1)