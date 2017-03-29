import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import numpy as np
NUM_CLASSES = 102

VGG_MEAN = [103.939, 116.779, 123.68]

class VGG16(object):

	def __init__(self, trainable=True, dropout=0.5):
		self.trainable = trainable
		self.dropout = dropout
		self.parameters = []

	def build(self, rgb, train_mode=None):

		"""
		:param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
		:param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
		"""

		rgb_scaled = rgb * 255.0

		red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
		assert red.get_shape().as_list()[1:] == [224, 224, 1]
		assert green.get_shape().as_list()[1:] == [224, 224, 1]
		assert blue.get_shape().as_list()[1:] == [224, 224, 1]
		bgr = tf.concat(axis=3, values=[
			blue - VGG_MEAN[0],
			green - VGG_MEAN[1],
			red - VGG_MEAN[2],
		])
		assert bgr.get_shape().as_list()[1:] == [224, 224, 3]


		# conv1_1
		with tf.name_scope('conv1_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
													 stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(bgr, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
								 trainable=True, name='biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv1_1 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]
			
		# conv1_2
		with tf.name_scope('conv1_2') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
													 stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
								trainable=True, name='biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv1_2 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

		self.pool1 = tf.nn.max_pool(self.conv1_2,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool1')

		with tf.name_scope('conv2_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
													 stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
								trainable=True, name='biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv2_1 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

		with tf.name_scope('conv2_2') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
													 stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
								 trainable=True, name='biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv2_2 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

		self.pool2 = tf.nn.max_pool(self.conv2_2,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool2')

		with tf.name_scope('conv3_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
													 stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
								trainable=True, name='biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv3_1 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

		with tf.name_scope('conv3_2') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
													 stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
								trainable=True, name='biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv3_2 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

		with tf.name_scope('conv3_3') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
													 stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
								trainable=True, name='biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv3_3 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

		self.pool3 = tf.nn.max_pool(self.conv3_3,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool3')

		# conv4_1
		with tf.name_scope('conv4_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
													 stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
								 trainable=True, name='biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv4_1 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

		# conv4_2
		with tf.name_scope('conv4_2') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
													 stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
								 trainable=True, name='biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv4_2 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

		# conv4_3
		with tf.name_scope('conv4_3') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
													 stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
								 trainable=True, name='biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv4_3 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

		# pool4
		self.pool4 = tf.nn.max_pool(self.conv4_3,
							   ksize=[1, 2, 2, 1],
							   strides=[1, 2, 2, 1],
							   padding='SAME',
							   name='pool4')


		# conv5_1
		with tf.name_scope('conv5_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
													 stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
								 trainable=True, name='biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv5_1 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

		# conv5_2
		with tf.name_scope('conv5_2') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
													 stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
								 trainable=True, name='biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv5_2 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

		# conv5_3
		with tf.name_scope('conv5_3') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
													 stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
								 trainable=True, name='biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv5_3 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

		# pool5
		self.pool5 = tf.nn.max_pool(self.conv5_3,
							   ksize=[1, 2, 2, 1],
							   strides=[1, 2, 2, 1],
							   padding='SAME',
							   name='pool4')

		# fc1
		with tf.name_scope('fc1') as scope:
			shape = int(np.prod(self.pool5.get_shape()[1:]))
			fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
														 dtype=tf.float32,
														 stddev=1e-1), name='weights')
			fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
								 trainable=True, name='biases')
			pool5_flat = tf.reshape(self.pool5, [-1, shape])
			fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
			self.fc1 = tf.nn.relu(fc1l)
			self.parameters += [fc1w, fc1b]

		# fc2
		with tf.name_scope('fc2') as scope:
			fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
														 dtype=tf.float32,
														 stddev=1e-1), name='weights')
			fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
								 trainable=True, name='biases')
			fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
			self.fc2 = tf.nn.relu(fc2l)
			self.parameters += [fc2w, fc2b]

		# fc3
		with tf.name_scope('fc3') as scope:
			fc3w = tf.Variable(tf.truncated_normal([4096, NUM_CLASSES],
														 dtype=tf.float32,
														 stddev=1e-1), name='weights')
			fc3b = tf.Variable(tf.constant(1.0, shape=[NUM_CLASSES], dtype=tf.float32),
								 trainable=True, name='biases')
			self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
			self.parameters += [fc3w, fc3b]        



	def loss(self, labels):
		labels = tf.to_int64(labels)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
			logits=self.fc3l, name='xentropy')
		return tf.reduce_mean(cross_entropy, name='xentropy_mean')


	def training(self, loss, learning_rate):
		tf.summary.scalar('loss', loss)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		
		self.global_step = tf.Variable(0, name='global_step', trainable=False)
		train_op = optimizer.minimize(loss, global_step=self.global_step)
		#train_op = optimizer.minimize(loss)

		return train_op
