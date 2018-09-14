# This module is used to unpack compressed data file downloaded from MNIST org net
# And provide methods to read (label, image) pairs
import os
import struct
import gzip
import numpy as np
import tensorflow as tf

MNIST_IMAGE_SIZE = 28		# MNIST:	28x28 image



def _parse_example(serialized_example):
	"""internal method to parse single TFRecord example, 
		This will be used as a map function for TFRecordDataset"""
	feature = tf.parse_single_example(serialized_example, 
			features={'label': tf.FixedLenFeature([], tf.float32),
					'image': tf.FixedLenFeature([], tf.string)})
	# Reinterpret the bytes of a string (from the file) as a vector of numbers.
	img = tf.decode_raw(feature['image'], tf.uint8)
	# reshape the image to proper shape
	img = tf.reshape(img, [28, 28, 1])
	# cast image data type to tf.float32 and normalize the image
	img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
	# return a tuple
	return feature['label'], img

# read a list of TFRecord files, return a batch of data
# we try to use tf.data api to do this
def inputs(fileNames, batchSize, shuffle):
	"""fileNames:	names of files"""
	# check validity of input file names
	for f in fileNames:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)
	
	with tf.name_scope('inputs'):
		# create a TFRecord dataset to read from input file names
		files = tf.data.TFRecordDataset(fileNames)
		# convert to a dataset consisting all example pairs from all input files
		# use 16 threads to do conversion
		dataset = files.map(map_func=_parse_example, num_parallel_calls=16)
		# convert to a dataset with shuffled examples if necessary
		# randomly sample each 1000 examples from old dataset
		if shuffle:
			dataset = dataset.shuffle(1000)
		# convert to a dataset with batched examples
		dataset = dataset.batch(batchSize)
		# add prefetch function to the dataset
		# we only prefetch 1 batch here, that's enough
		dataset = dataset.prefetch(1)
		# use iterator to consume batches
		it = dataset.make_one_shot_iterator()
		labels, images = it.get_next()
		# reshape labels from 2D tensor ([batchSize,1]) to 1D tensor ([batchSize])
		return tf.reshape(labels, [batchSize]), images
		