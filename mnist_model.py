#coding:utf-8
import tensorflow as tf
import mnist_input
import time
import re


# helper to create summary for a tensor op
def _activation_summary(x):
	"""x is an input tensor, the op connected to x will be summarized."""
	# record tensor op name (excluding the device scope name)
	tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
	# add histogram of tensor x to the summary
	tf.summary.histogram(tensor_name + '/activations', x)
	tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

# create a variable on host side
def _variable_on_host(name, shape, dtype, initializer):
	"""initializer is used for variable initialization."""
	# create host context
	with tf.device('/cpu:0'):
		var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
	return var

# create a variable tensor with given shape on host side
def weight_variable(name, shape):
	"""initialized with a truncated normal distribution."""
	return _variable_on_host(name, shape, tf.float32, 
						initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))

# create a variable tensor with given shape on host side
def bias_variable(name, shape):
	"""initialized with a constant value."""
	return _variable_on_host(name, shape, tf.float32, 
						initializer=tf.constant_initializer(value=0, dtype=tf.float32))

# forward propagation (inference the logits)
def inference(images):
	"""images: a batch of images returned by input pipe."""
	# 1st 2D convolution
	with tf.variable_scope('conv1'):
		# get conv kernel on host side
		kernel = weight_variable(name='weight', shape=[5,5,1,32])
		biases = bias_variable(name='biases', shape=[32])
		# perform conv
		conv = tf.nn.conv2d(images, kernel, strides=[1,1,1,1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		# activate
		conv1 = tf.nn.relu(pre_activation)
		# record the activation
		_activation_summary(conv1)
		
	# max pooling
	pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')
	
	# 2nd 2D convolution
	with tf.variable_scope('conv2'):
		# get conv kernel on host side
		kernel = weight_variable(name='weight', shape=[5,5,32,64])
		biases = bias_variable(name='biases', shape=[64])
		# perform conv
		conv = tf.nn.conv2d(pool1, kernel, strides=[1,1,1,1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		# activate
		conv2 = tf.nn.relu(pre_activation)
		# record this activation
		_activation_summary(conv2)

	# max pooling
	pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')
	
	# full connection
	# reshape the feature map
	pool2 = tf.reshape(pool2, [-1, 7 * 7 * 64])
	with tf.variable_scope('full_connection'):
		weights = weight_variable(name='weight', shape=[7*7*64, 1024])
		biases = bias_variable(name='biases', shape=[1024])
		pre_activation = tf.nn.bias_add(tf.matmul(pool2, weights), biases)
		# activate
		fc = tf.nn.relu(pre_activation)
		# record this activation
		_activation_summary(fc)
		
	with tf.variable_scope('outputs'):
		weights = weight_variable(name='weight', shape=[1024, 10])
		biases = bias_variable(name='biases', shape=[10])
		softmax_linear = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc,weights), biases))
		# record the softmax result
		_activation_summary(softmax_linear)
	
	# We don't apply softmax here because
	# tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
	# and performs the softmax internally for efficiency.
	return softmax_linear

# define loss function
def loss(logits, labels):
	"""logits is the tensor returned by inference.
		labels is a batch of labels returned by inputs."""
	labels = tf.cast(labels, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
				labels=labels, logits=logits, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	# save local cross entropy to global collection
	tf.add_to_collection('losses', cross_entropy_mean)
	# get loss
	return tf.add_n(tf.get_collection('losses'), name='total_loss')
	
# compute and summarize losses (from different devices) and total loss moved average
def _add_loss_summaries(total_loss):
 	# create a tensorflow Moving Average manager with relaxation = 0.9
	loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
	# get all loss variables
	losses = tf.get_collection('losses')
	# apply the manager to losses and total loss
	loss_averages_op = loss_averages.apply(losses + [total_loss])
	# add summaries
	for l in losses + [total_loss]:
		# record raw values
		tf.summary.scalar(l.op.name + ' (raw)', l)
		# record averaged values
		tf.summary.scalar(l.op.name, loss_averages.average(l))
	# over
	return loss_averages_op

# use loss info to train model parameters
def train(total_loss, learning_rate, global_step):
	"""a train op will be returned to adjust all variables in the model."""
	"""add dynamically decaying learning rate."""
	
	# compute moving averaged losses
	loss_averages_op = _add_loss_summaries(total_loss)
	# use moving average to :
	with tf.control_dependencies([loss_averages_op]):	
		# compute gradient descent
		opt = tf.train.GradientDescentOptimizer(learning_rate)
		grads = opt.compute_gradients(total_loss)
	
	# now we can apply gradient info
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
	
	# Add histograms for trainable variables.
	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)
	# Add histograms for gradients.
	for grad, var in grads:
		if grad is not None:
			tf.summary.histogram(var.op.name + '/gradients', grad)
			
	# Track the moving averages of all trainable variables.
	# create the manager
	variable_averages = tf.train.ExponentialMovingAverage(0.9, global_step)
	# make sure to average variables after graph updated
	with tf.control_dependencies([apply_gradient_op]):
		variables_averages_op = variable_averages.apply(tf.trainable_variables())
	# over
	return variables_averages_op

