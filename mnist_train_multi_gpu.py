from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

import tensorflow as tf
import mnist_input
import mnist_model

FILE_NAMES = './mnist-train-tfrecord'

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
		loss_name = re.sub('%s_[0-9]*/' % mnist_model.TOWER_NAME, '', l.op.name)
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
			expanded_g = tf.expand_dims(g, 0)
			# Append on a 'tower' dimension which we will average over below.
      			grads.append(expanded_g)

		# Average over the 'tower' dimension.
		# tf.concat: concatinate input in a given axis
		grad = tf.concat(axis=0, values=grads)
		# tf.reduce_mean: compute the average along given axis
		grad = tf.reduce_mean(grad, 0)
# 2018/09/07
    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def train():
	"""Train MNIST for a number of steps."""
  	# create a new graph and use it as default graph in the following context:
	with tf.Graph().as_default():
		global_step = tf.train.get_or_create_global_step()

		# Get images and labels
		# Force input pipeline to CPU:0 to avoid operations sometimes ending up on
		# GPU and resulting in a slow down.
		with tf.device('/cpu:0'):
			labels, images = mnist_input.inputs([FILE_NAMES], batchSize=100, shuffle=True)
	
		# Build a Graph that computes the logits predictions from the
		# inference model.
		logits = mnist_model.inference(images)
	
		# Calculate loss.
		loss = mnist_model.loss(logits, labels)
	
		# Build a Graph that trains the model with one batch of examples and
		# updates the model parameters.
		train_op = mnist_model.train(loss, 0.001, global_step)
	
		class _LoggerHook(tf.train.SessionRunHook):
		"""Logs loss and runtime."""
			def begin(self):
				self._step = -1
				self._start_time = time.time()
		
			def before_run(self, run_context):
				self._step += 1
				return tf.train.SessionRunArgs(loss)  # Asks for loss value.
		
			def after_run(self, run_context, run_values):
				if self._step % 100 == 0:
					current_time = time.time()
					duration = current_time - self._start_time
					self._start_time = current_time
		
					loss_value = run_values.results
					sec_per_batch = float(duration / 100)
		
					format_str = ('%s: step %d, loss = %.2f (%.3f sec/batch)')
					print (format_str % (datetime.now(), self._step, loss_value, sec_per_batch))
	
		with tf.train.MonitoredTrainingSession(hooks=[_LoggerHook()]) as mon_sess:
			while not mon_sess.should_stop():
				mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument
	train()


if __name__ == '__main__':
  tf.app.run()