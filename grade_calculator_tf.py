import tensorflow as tf
import numpy as np

# Neural Network
INPUT_LAYER_SIZE = 2
OUTPUT_LAYER_SIZE = 1
HIDDEN_LAYER_SIZE = 3

def weight_variable(i, j):
	# initial = tf.truncated_normal([INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE], stddev=0.1)
	initial = np.random.randn(i, j).astype(np.float32)
	return tf.Variable(initial)

W1 = weight_variable(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE)
W2 = weight_variable(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE)

x = tf.placeholder(tf.float32, [None, 2])

z2 = tf.matmul(x, W1)
a2 = tf.sigmoid(z2)

z3 = tf.matmul(a2, W2)
yHat = tf.sigmoid(z3)

y = tf.placeholder(tf.float32, shape=[None, 1])


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(z3, y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
# Evaluate model
correct_pred = tf.equal(tf.argmax(z3,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()


# Training data
train_x = np.array(([3,5], [5,1], [10,2]), dtype=float)
train_y = np.array(([75], [82], [93]), dtype=float)

# Normalize
train_x= train_x/np.amax(train_x, axis=0)
train_y = train_y/100 #Max test score is 100

# Testing data
test_x = np.array(([8,8], [5,1], [10,8]), dtype=float)
test_x= test_x/np.amax(test_x, axis=0)


with tf.Session() as sess:
	sess.run(init)
	
	#for i in range(100):
	#	sess.run(train_step, feed_dict={x: train_x, y_: train_y})
	for i in range(100):
		sess.run(optimizer, feed_dict={x: train_x, y: train_y})
	# Calculate batch accuracy
	acc = sess.run(accuracy, feed_dict={x:  train_x, y:  train_y})
	# Calculate batch loss
	loss = sess.run(cost, feed_dict={x:  train_x, y:  train_y})
	print("Accuracy: " + str(acc))
	print("Loss: " + str(loss))

	# evaluation
	#train_accuracy = accuracy.eval(feed_dict={x: train_x, y_: train_y})
	#print("training accuracy %g"%(train_accuracy))

	print(sess.run(yHat, feed_dict={x: train_x}))