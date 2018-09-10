# This module is used to unpack compressed data file downloaded from MNIST org net
# And provide methods to read (label, image) pairs
import os
import struct
import gzip
import numpy as np
import tensorflow as tf

MNIST_IMAGE_SIZE = 28		# MNIST:	28x28 image


# method to unpack compressed MNIST raw data if necessary
def unpack_mnist_data(path):
	"""path: directory of raw data."""
	# check if the path exists
	if not os.path.exists(path):
		return
	# check if target raw data exist
	trainImgFileName = 'train-images-idx3-ubyte.gz'
	trainLabFileName = 'train-labels-idx1-ubyte.gz'
	evalImgFileName = 't10k-images-idx3-ubyte.gz'
	evalLabFileName = 't10k-labels-idx1-ubyte.gz'
	if not (os.path.exists(os.path.join(path, trainImgFileName)) and
		  os.path.exists(os.path.join(path, trainLabFileName)) and
		  os.path.exists(os.path.join(path, evalImgFileName)) and
		  os.path.exists(os.path.join(path, evalLabFileName))):
		  return
	# unpack compressed data
	nameList = [trainImgFileName, trainLabFileName, evalImgFileName, evalLabFileName]
	for i in range(4):
		if os.path.exists(nameList[i].replace('.gz', '')):
			continue
		with gzip.GzipFile(os.path.join(path, nameList[i])) as gz_file:
			with open(nameList[i].replace('.gz',''), 'w') as unzip_file:
				unzip_file.write(gz_file.read())
	# over
	return

# convert unpacked MNIST data to TFRecord file format
def convert_to_tfrecord(path):
	"""path:	the directory of unpacked data file."""
	nameList = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
				't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']
	# convert train data
	with open(os.path.join(path, 'train-labels-idx1-ubyte')) as trainLabFile:
		# read label file header
		magic, n = struct.unpack('>II', trainLabFile.read(8))
		# read label data into buffer
		labBuf = struct.unpack('>'+str(n)+'B', trainLabFile.read())
		
	with open(os.path.join(path, 'train-images-idx3-ubyte')) as trainImgFile:
		# read image file header
		magic, n, rows, cols = struct.unpack('>IIII', trainImgFile.read(16))
		# read image data into buffer
		imgBuf = struct.unpack('>'+str(n*rows*cols)+'B', trainImgFile.read())
		imgBuf = np.reshape(imgBuf, [n, rows*cols]).astype(np.uint8)
	
	# generate tfrecord examples
	with tf.python_io.TFRecordWriter('mnist-train-tfrecord') as record_writer:
		for i in range(n):
			__lab = np.array(labBuf[i], dtype=np.uint8).astype(np.float32)
			__img = imgBuf[i,:]		# read the i-th item of image buffer
			tfrecord_feature = {}	# create an empty dict
			# add one feature 'label' to the dict
			# note: the value arg must be an TF tensor, so we use [] to convert from numpy arrays
			tfrecord_feature['label'] = tf.train.Feature(float_list=tf.train.FloatList(value=[__lab])
			# add another feature 'image' to the dict
			# note: the BytesList method must recieve a string tensor as input
			# so we use numpy array's tobytes() to convert
			tfrecord_feature['image'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[__img.tobytes()])
			# combine all features together to a 'Features' and use it to create an example
			# the tf.train.Features(feature=...) feature arg is used as an positional arg and cannot
			# be omitted
			example = tf.train.Example(features=tf.train.Features(feature=tfrecord_feature))
			record_writer.write(example.SerializeToString())
	
	# convert eval data
	with open(os.path.join(path, 't10k-labels-idx1-ubyte')) as trainLabFile:
		# read label file header
		magic, n = struct.unpack('>II', trainLabFile.read(8))
		# read label data into buffer
		labBuf = struct.unpack('>'+str(n)+'B', trainLabFile.read())
		
	with open(os.path.join(path, 't10k-images-idx3-ubyte')) as trainImgFile:
		# read image file header
		magic, n, rows, cols = struct.unpack('>IIII', trainImgFile.read(16))
		# read image data into buffer
		imgBuf = struct.unpack('>'+str(n*rows*cols)+'B', trainImgFile.read())
		imgBuf = np.reshape(imgBuf, [n, rows*cols]).astype(np.uint8)
	
	# generate tfrecord examples
	with tf.python_io.TFRecordWriter('mnist-eval-tfrecord') as record_writer:
		for i in range(n):
			__lab = np.array(labBuf[i], dtype=np.uint8).astype(np.float32)
			__img = imgBuf[i,:]
			tfrecord_feature = {}
			tfrecord_feature['label'] = tf.train.Feature(float_list=tf.train.FloatList(value=[__lab])
			tfrecord_feature['image'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[__img.tobytes()])
			example = tf.train.Example(features=tf.train.Features(feature=tfrecord_feature))
			record_writer.write(example.SerializeToString())	
	# over
	return

# read single example from TFRecord
def read_from_tfrecord(fileNameQ):
	"""read one (label, image) pair from a TFRecord file.
		the label will be float32, image will be 28x28x1 one-channel float32 tensor."""
	# define internal method to parse single example
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
	
	# read from file name queue
	# create a TFRecord queue runner
	reader = tf.TFRecordReader()
	# read (filename(not needed), example) pair
	_, serialized_example = reader.read(fileNameQ)
	# parse the example and return
	return _parse_example(serialized_example)

# create a data batch, using single examples
def _generate_batch(label, image, minQSize, batchSize, shuffle):
	"""generate a batch of (label, image) data.
		shuffle controls the shuffle behavior."""
	# threads used for parallel
	num_preprocess_threads = 16
	# shuffle the data
	if shuffle:
		labels, images = tf.train.shuffle_batch([label, image], 
									batch_size=batchSize,
									num_threads=num_preprocess_threads,
									capacity=minQSize + 3 * batchSize,
									min_after_dequeue = minQSize)
	else:
		labels, images = tf.train.batch([label, image],
								batch_size=batchSize,
								num_threads=num_preprocess_threads,
								capacity=minQSize + 3 * batchSize)
	# reshape labels from 2D tensor ([batchSize,1]) to 1D tensor ([batchSize])
	return tf.reshape(labels, [batchSize]), images

# read a list of TFRecord files, return a batch of data
def inputs(fileNames, batchSize, shuffle):
	"""fileNames:	names of files"""
	# check validity of input file names
	for f in fileNames:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)
	
	with tf.name_scope('inputs'):
		# create a file name queue
		fileNameQ = tf.train.string_input_producer(fileNames)
		# read single example from input files (single-item-pipe)
		label, image = read_from_tfrecord(fileNameQ)
		# generate a batch
		batch = _generate_batch(label, image, minQSize=int(0.1*batchSize), 
							batchSize=batchSize, shuffle=shuffle)
	
	# over
	return batch
		