import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import numpy as np
NUM_CLASSES = 10
beta = 0.001
TEMP_SOFTMAX = 3.0
VGG_MEAN = [103.939, 116.779, 123.68]

class Student(object):

	def __init__(self, trainable=True, dropout=0.5):
		self.trainable = trainable
		self.dropout = dropout
		self.parameters = []

	def build(self, rgb, train_mode=None):

		# conv1_1
		with tf.name_scope('student_conv1_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([5,5,3,64], dtype = tf.float32, stddev = 1e-2), name='weights', trainable = True)
                        conv = tf.nn.conv2d(rgb, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape = [64], dtype = tf.float32), name='biases', trainable = True)
			out = tf.nn.bias_add(conv, biases)
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))

			self.conv1_1 = tf.nn.relu(batch_norm, name=scope)
			self.parameters += [kernel, biases]
                self.pool1 = tf.nn.max_pool(self.conv1_1,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME',
                                name='pool1')


		with tf.name_scope('student_conv2_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([5,5,64,128]), name='weights', trainable = True)
			conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape = [128], dtype = tf.float32), name='biases', trainable = True)
			out = tf.nn.bias_add(conv, biases)
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv2_1 = tf.nn.relu(batch_norm, name=scope)
			self.parameters += [kernel, biases]
                self.pool2 = tf.nn.max_pool(self.conv2_1,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME',
                                name='pool2')

		# fc1
		with tf.name_scope('student_fc1') as scope:
			shape = int(np.prod(self.pool2.get_shape()[1:]))
			fc1w = tf.Variable(tf.truncated_normal([shape,1024]), name='weights', trainable = True)
			fc1b = tf.Variable(tf.constant(1.0, shape = [1024], dtype = tf.float32), name='biases', trainable = True)
			pool2_flat = tf.reshape(self.pool2, [-1, shape])
			fc1 = tf.nn.bias_add(tf.matmul(pool2_flat, fc1w), fc1b)
                        mean, var = tf.nn.moments(fc1, axes=[0])
                        batch_norm = (fc1 - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.fc1 = tf.nn.relu(batch_norm)
                        
			self.parameters += [fc1w, fc1b]


		# fc2
		with tf.name_scope('student_fc2') as scope:
			fc2w = tf.Variable(tf.truncated_normal([1024, NUM_CLASSES],
														 dtype=tf.float32, stddev=1e-2), name='weights', trainable = True)
			fc2b = tf.Variable(tf.constant(1.0, shape=[NUM_CLASSES], dtype=tf.float32),
								  name='biases', trainable = True)
			self.fc2 = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
                        mean, var = tf.nn.moments(self.fc2, axes=[0])
                        batch_norm = (self.fc2 - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.fc2 = tf.nn.relu(batch_norm)
			self.parameters += [fc2w, fc2b]  
                logits_temp = tf.divide(self.fc2, tf.constant(TEMP_SOFTMAX))

                return self.fc2

        def loss(self, labels):
                labels = tf.to_int64(labels)

                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=self.fc2, name='xentropy')
                return tf.reduce_mean(cross_entropy, name='xentropy')

	def training(self, loss,global_step,learning_rate):
		tf.summary.scalar('loss', loss)

		optimizer = tf.train.AdamOptimizer(learning_rate)
		
		train_op = optimizer.minimize(loss, global_step = global_step)

                return train_op
       
