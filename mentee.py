import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import pdb
import numpy as np
NUM_CLASSES = 102
TEMP_SOFTMAX = 5.0
VGG_MEAN = [103.939, 116.779, 123.68]

class VGG16Mentee(object):

	def __init__(self, trainable=True, dropout=0.5):
		self.trainable = trainable
		self.dropout = dropout
		self.parameters = []

	def build(self, rgb, train_mode=None):

		# conv1_1
		with tf.name_scope('mentee_conv1_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
													 stddev=1e-2), trainable = True, name='mentee_weights')
			conv = tf.nn.conv2d(rgb, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
								 trainable=True, name='mentee_biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv1_1 = tf.nn.relu(out, name=scope)
                        #pdb.set_trace() 
			self.parameters += [kernel, biases]
			
		self.pool1 = tf.nn.max_pool(self.conv1_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool1')
                
                #conv2_1
		with tf.name_scope('mentee_conv2_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
													 stddev=1e-2), trainable = True,name='mentee_weights')
			conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
								trainable=True, name='mentee_biases')
                
			out = tf.nn.bias_add(conv, biases)
			self.conv2_1 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

		self.pool2 = tf.nn.max_pool(self.conv2_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool2')
                #conv3_1
		with tf.name_scope('mentee_conv3_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
													 stddev=1e-2), trainable = True, name='mentee_weights')
			conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
								trainable=True, name='mentee_biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv3_1 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

		self.pool3 = tf.nn.max_pool(self.conv3_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool3')

		with tf.name_scope('mentee_conv4_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
													 stddev=1e-2), trainable = True, name='mentee_weights')
			conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
								trainable=True, name='mentee_biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv4_1 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

		self.pool4 = tf.nn.max_pool(self.conv4_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool4')

		with tf.name_scope('mentee_conv5_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
													 stddev=1e-2), trainable = True, name='mentee_weights')
			conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
								trainable=True, name='mentee_biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv5_1 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

		self.pool5 = tf.nn.max_pool(self.conv5_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool5')

		# fc1
		with tf.name_scope('mentee_fc1') as scope:
			shape = int(np.prod(self.pool5.get_shape()[1:]))
			fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
														 dtype=tf.float32, stddev=1e-2), trainable = True,name='mentee_weights')
			fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
								 trainable=True, name='mentee_biases')
			pool5_flat = tf.reshape(self.pool5, [-1, shape])
			fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
			self.fc1 = tf.nn.relu(fc1l)
                        self.fc1 = tf.nn.dropout(self.fc1, 0.5)
			self.parameters += [fc1w, fc1b]

		with tf.name_scope('mentee_fc2') as scope:
			fc2w = tf.Variable(tf.truncated_normal([4096, NUM_CLASSES],
														 dtype=tf.float32, stddev=1e-2), trainable = True,name='mentee_weights')
			fc2b = tf.Variable(tf.constant(1.0, shape=[NUM_CLASSES], dtype=tf.float32),
								 trainable=True, name='mentee_biases')
			self.fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
			self.parameters += [fc2w, fc2b]

                logits_temp = tf.divide(self.fc2l, tf.constant(TEMP_SOFTMAX))
            
                return self.pool4, self.conv5_1, self.fc2l, tf.nn.softmax(logits_temp)

	def mentee_loss(self, labels):
		labels = tf.to_int64(labels)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
			logits=self.fc2l, name='xentropy')
		return tf.reduce_mean(cross_entropy, name='xentropy_mean')


	def training(self, loss, learning_rate, global_step):
		tf.summary.scalar('loss', loss)
		optimizer = tf.train.AdamOptimizer(learning_rate)
		
		#self.global_step = tf.Variable(0, name='global_step', trainable=False)
		train_op = optimizer.minimize(loss, global_step=global_step)
		

		return train_op
