import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import pdb
import numpy as np
NUM_CLASSES = 102
TEMP_SOFTMAX = 5.0
VGG_MEAN = [103.939, 116.779, 123.68]

class VGG16Mentor(object):

	def __init__(self, trainable=True, dropout=0.5):
		self.trainable = trainable
		self.dropout = dropout
		self.parameters = []

	def build(self, rgb, train_mode=None):

		# conv1_1
		with tf.name_scope('mentor_conv1_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
													 stddev=1e-2), trainable = False, name='mentor_weights')
			conv = tf.nn.conv2d(rgb, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
								 trainable=False, name='mentor_biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv1_1 = tf.nn.relu(out, name=scope)
                        #pdb.set_trace() 
			self.parameters += [kernel, biases]
			
		self.pool1 = tf.nn.max_pool(self.conv1_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool1')
                #pdb.set_trace()
                #conv2_1
		with tf.name_scope('mentor_conv2_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
													 stddev=1e-2), trainable = False,name='mentor_weights')
			conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
								trainable=False, name='mentor_biases')
                 #       pdb.set_trace()
			out = tf.nn.bias_add(conv, biases)
			self.conv2_1 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

		self.pool2 = tf.nn.max_pool(self.conv2_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool2')
                #conv3_1
		with tf.name_scope('mentor_conv3_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
													 stddev=1e-2), trainable = False, name='mentor_weights')
			conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
								trainable=False, name='mentor_biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv3_1 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

		self.pool3 = tf.nn.max_pool(self.conv3_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool3')


		# conv4_1
		with tf.name_scope('mentor_conv4_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
													 stddev=1e-2), trainable = False,name='mentor_weights')
			conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
								 trainable=False, name='mentor_biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv4_1 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

		self.pool4 = tf.nn.max_pool(self.conv4_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool4')
		# conv5_1
		with tf.name_scope('mentor_conv5_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
													 stddev=1e-2), trainable = False, name='mentor_weights')
			conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
								 trainable=False, name='mentor_biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv5_1 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

		self.pool5 = tf.nn.max_pool(self.conv5_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool5')
		# fc1
		with tf.name_scope('mentor_fc1') as scope:
			shape = int(np.prod(self.pool5.get_shape()[1:]))
			fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
														 dtype=tf.float32, stddev=1e-2), trainable = False,name='mentor_weights')
			fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
								 trainable=False, name='mentor_biases')
			pool5_flat = tf.reshape(self.pool5, [-1, shape])
			fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
			self.fc1 = tf.nn.relu(fc1l)
                        self.fc1 = tf.nn.dropout(self.fc1, 0.5)
			self.parameters += [fc1w, fc1b]

		with tf.name_scope('mentor_fc2') as scope:
			fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
														 dtype=tf.float32, stddev=1e-2), trainable = False,name='mentor_weights')
			fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
								 trainable=False, name='mentor_biases')
			fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
			self.fc2 = tf.nn.relu(fc2l)
                        self.fc2 = tf.nn.dropout(self.fc2, 0.5)
			self.parameters += [fc2w, fc2b]

		with tf.name_scope('mentor_fc3') as scope:
			fc3w = tf.Variable(tf.truncated_normal([4096, NUM_CLASSES],
														 dtype=tf.float32, stddev=1e-2), trainable = False,name='mentor_weights')
			fc3b = tf.Variable(tf.constant(1.0, shape=[NUM_CLASSES], dtype=tf.float32),
								 trainable=False, name='mentor_biases')
			fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
			self.fc3 = tf.nn.relu(fc3l)
                        self.fc3 = tf.nn.dropout(self.fc3, 0.5)
			self.parameters += [fc3w, fc3b]
                
                logits_temp = tf.divide(self.fc3, tf.constant(TEMP_SOFTMAX))
            
                return self.conv3_1, self.conv5_1, self.fc3, tf.nn.softmax(logits_temp)

	def mentor_loss(self, labels):
		labels = tf.to_int64(labels)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
			logits=self.fc3, name='xentropy')
		return tf.reduce_mean(cross_entropy, name='xentropy_mean')


	def training(self, loss, learning_rate):
		tf.summary.scalar('loss', loss)
		optimizer = tf.train.AdamOptimizer(learning_rate)
		
		self.global_step = tf.Variable(0, name='global_step', trainable=False)
		train_op = optimizer.minimize(loss, global_step=self.global_step)
		

		return train_op
