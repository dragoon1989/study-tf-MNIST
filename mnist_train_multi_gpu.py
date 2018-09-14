from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time
import numpy as np

import tensorflow as tf
import mnist_input
import mnist_model

FILE_NAMES = './mnist-train-tfrecord'
NUM_GPU = 2

# calculate the local loss for current batch on current device
# this method should be called in context of each device
def tower_loss(scope, images, labels):
	"""scope:	name scope asigned to current device."""
	"""images and labels:	data batch."""
	
	# construct network on current device
	logits = mnist_model.inference(images)
	# construct backward projection (i.e. compute loss) on current device
	# we must stop the data flow before the return node
	# to avoid 'fetch-op' from data node outside current device
	_ = mnist_model.loss(logits, labels)
	# now we use another method to fetch data only from current device
	# and compute the local loss
	losses = tf.get_collection('losses', scope)
	# compute total loss
	total_loss = tf.add_n(losses, name='total_loss')
	# add summaries
	for l in losses + [total_loss]:
		loss_name = re.sub('%s_[0-9]*/' % 'tower', '', l.op.name)
		tf.summary.scalar(loss_name, l)
	
	# over, return the total loss for current batch
	return total_loss

# method to average gradients from different devices
def average_gradients(tower_grads):
	"""tower_grads is a list of the gradients from all devices."""
	average_grads = []
	for grad_and_vars in zip(*tower_grads):
		# the python zip(*) method: merge elements with same idx in cells and return
		# Note that each grad_and_vars looks like the following:
		# ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
		grads = []
		for g, _ in grad_and_vars:
			# we only need the grad info (g)
			# Add 0 dimension to the gradients to represent the tower.
			# tf.expand_dims: insert a dimension with length 1 to a given axis
			# the expanded_g looks like: extended_g = [g]
			expanded_g = tf.expand_dims(g, 0)
			# successively append gradient info to the container 'grads'
			# the grads looks like: grads = [[g0], [g1], [g2], ...]
      		grads.append(expanded_g)

		# Average over the 'tower' dimension.
		# tf.concat: concatinate input in a given axis (here is the 0-th axis)
		# the grad looks like: grad = [g0, g1, g2, ...]
		grad = tf.concat(axis=0, values=grads)
		# tf.reduce_mean: compute the average along given axis
		# so the grad will be averaged to: grad = mean(g0,g1,g2,...)
		grad = tf.reduce_mean(grad, 0)

	# Keep in mind that the Variables are redundant because they are shared
	# across towers. So .. we will just return the first tower's pointer to
	# the Variable.
	# i.e. var0_gpu0 = var0_gpu1 = var0_gpu2 = ...
	v = grad_and_vars[0][1]
	grad_and_var = (grad, v)
	average_grads.append(grad_and_var)
	return average_grads

def train():
	"""Train MNIST for a number of steps."""
  	# create a new graph and use it as default graph in the following context:
	# this context will also be created on host side
	with tf.Graph().as_default(), tf.device('/cpu:0'):
		#global_step = tf.train.get_or_create_global_step()	# why not use this?
		global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

		# Create an optimizer that performs gradient descent.
		opt = tf.train.GradientDescentOptimizer(0.001)

		# create data batch on host side
		labels, images = mnist_input.inputs([FILE_NAMES], batchSize=100, shuffle=True)
		batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([labels, images], capacity=2 * NUM_GPU)

		# compute gradients on each device
		tower_grads = []

		with tf.variable_scope(tf.get_variable_scope()):
			for i in xrange(NUM_GPU):
				with tf.device('/gpu:%d' % i):
					with tf.name_scope('%s_%d' % ('tower', i)) as scope:
						# Dequeues one batch for the GPU
						label_batch, image_batch = batch_queue.dequeue()
						# compute local loss for the batch
						loss = tower_loss(scope, image_batch, label_batch)
						# share the variables of model from gpu:0
						tf.get_variable_scope().reuse_variables()
						# Retain the summaries from the final tower.
						summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
						# Calculate the gradients for the batch of data on this device.
						grads = opt.compute_gradients(loss)
						# push local gradients into the global container 'tower_grads'
						tower_grads.append(grads)
	
		# We must calculate the mean of each gradient. Note that this is the
		# synchronization point across all towers.
		grads = average_gradients(tower_grads)
		
		# Add histograms for gradients.
		for grad, var in grads:
			if grad is not None:
				summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
		
		# Apply the gradients to adjust the shared variables.
		apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

		# Add histograms for trainable variables.
		for var in tf.trainable_variables():
			summaries.append(tf.summary.histogram(var.op.name, var))
		
		# Track the moving averages of all trainable variables.
		variable_averages = tf.train.ExponentialMovingAverage(0.9, global_step)
		variables_averages_op = variable_averages.apply(tf.trainable_variables())

		# Group all updates to into a single train op.
		train_op = tf.group(apply_gradient_op, variables_averages_op)

		# Create a saver to save all variables
		saver = tf.train.Saver(tf.global_variables())

		# Build the summary operation from the last tower summaries.
		summary_op = tf.summary.merge(summaries)

		# Build an initialization operation to run below.
		init = tf.global_variables_initializer()

		# Start running operations on the Graph. allow_soft_placement must be set to
		# True to build towers on GPU, as some of the ops do not have GPU
		# implementations.
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		# First of all, initialize the model variables
		sess.run(init)

		# Then, start the queue runners.
		tf.train.start_queue_runners(sess=sess)
		# run the model N times
		for step in xrange(50000):
			start_time = time.time()
			_, loss_value = sess.run([train_op, loss])
			duration = time.time() - start_time

			assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

			if step % 10 == 0:
				num_examples_per_step = 100/2
				examples_per_sec = num_examples_per_step / duration
				sec_per_batch = duration / 2
				format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
				print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))


def main(argv=None):  # pylint: disable=unused-argument
	train()


if __name__ == '__main__':
  tf.app.run()