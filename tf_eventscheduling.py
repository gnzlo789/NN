import tensorflow as tf
import skflow
from sklearn.metrics import accuracy_score
import numpy

# Data
x_train = numpy.array([
			[3,3,2,15,4],
			[2,0,2,13,3],
			[3,1,3,12,3],
			[0,3,3,12,4],
			[1,2,4,10,4],
			[3,3,5,9,3],
			[3,2,4,13,4],
			[1,2,2,15,4],
			[1,2,4,11,4],
			[3,2,5,8,4],
			[0,1,1,16,4],
			[4,4,3,14,3],
			[1,0,4,11,3],
			[1,1,2,14,3],
			[3,3,4,12,3]])
y_train = numpy.array([
			[0,1],
			[0,1],
			[0,1],
			[0,1],
			[1,0],
			[1,0],
			[0,1],
			[0,1],
			[1,0],
			[1,0],
			[0,1],
			[0,1],
			[1,0],
			[0,1],
			[0,1]])

x_test = numpy.array([
			[1,0,2,8,3], # AMP 0
			[1,2,3,13,4], # AMP 1
			[1,0,1,12,4], # AMP 1
			[4,3,2,15,3], # AMP 1
			[3,3,5,9,4]])
y_test = numpy.array([
			[1,0],
			[0,1],
			[0,1],
			[0,1],
			[1,0]])

# Parameters
learning_rate = 0.05
training_epochs = 500
batch_size = 25
display_step = 100

# Network Parameters
n_input = 5 # data input
n_classes = 2
n_hidden_1 = 10 # 1st layer num features
n_hidden_2 = 20 # 2nd layer num features
# dropout = 0.75

# Input
x = tf.placeholder(tf.float32, [None, 5])
y = tf.placeholder(tf.float32, [None, n_classes])

# Create model
def mlp(_x, _weights, _biases, act=tf.nn.relu):
	layer1 = act( tf.add(tf.matmul(_x, _weights['h1']), _biases['b1']) )
	layer2 = act( tf.add(tf.matmul(layer1, _weights['h2']), _biases['b2']) )
	return tf.add(tf.matmul(layer2, _weights['out']), _biases['out'])

weights = { # Hay que probar con stddev=0.2 -> 1.0/n_input
	'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1)),
	'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
	'out': tf.Variable(tf.truncated_normal([n_hidden_2, n_classes], stddev=0.1))
}

biases = {
	'b1': tf.Variable(tf.zeros([n_hidden_1])),
	'b2': tf.Variable(tf.zeros([n_hidden_2])),
	'out': tf.Variable(tf.zeros([n_classes]))
}

# Construct model
pred = mlp(x, weights, biases) #act=tf.sigmoid

# Define loss and optimizer
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(pred, y) )
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)

	avg_cost = 0
	total_batch = 15
	for epoch in range(training_epochs):
		batch_xs = x_train
		batch_ys = y_train
		sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
		avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / total_batch

		# Display logs per epoch step
		if epoch % display_step == 0:
			print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
	
	print "Optimization Finished!"
	
	# Test model
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	# Calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print "Accuracy:", accuracy.eval({x: x_test, y: y_test})
	print(sess.run(pred, feed_dict={x: x_test}))




x_train = numpy.array([
			[3,3,2,15,4],
			[2,0,2,13,3],
			[3,1,3,12,3],
			[0,3,3,12,4],
			[1,2,4,10,4],
			[3,3,5,9,3],
			[3,2,4,13,4],
			[1,2,2,15,4],
			[1,2,4,11,4],
			[3,2,5,8,4],
			[0,1,1,16,4],
			[4,4,3,14,3],
			[1,0,4,11,3],
			[1,1,2,14,3],
			[3,3,4,12,3]], dtype=float)

y_train = numpy.array([
			[1],
			[1],
			[1],
			[1],
			[0],
			[0],
			[1],
			[1],
			[0],
			[0],
			[1],
			[1],
			[0],
			[1],
			[1]])

x_test = numpy.array([
			[1,0,2,8,3], # AMP 0
			[1,2,3,13,4], # AMP 1
			[1,0,1,12,4], # AMP 1
			[4,3,2,15,3], # AMP 1
			[3,3,5,9,4]], dtype=float)

y_test = numpy.array([
			[0],
			[1],
			[1],
			[1],
			[0]])


classifier = skflow.TensorFlowDNNClassifier(
	hidden_units=[10, 20, 10], 
	n_classes=2, 
	batch_size=5, 
	steps=500,
	optimizer='Adam',
	learning_rate=0.05)
classifier.fit(x_train, y_train)
print(accuracy_score(classifier.predict(x_test), y_test))