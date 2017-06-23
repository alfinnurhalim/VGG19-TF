import tensorflow as tf 
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import pdb
import numpy as np
NUM_CLASSES = 10
VGG_MEAN = [103.939, 116.779, 123.68]

class Teacher(object):

	def __init__(self, trainable=True, dropout=0.5):
		self.trainable = trainable
		self.dropout = dropout
		self.parameters = []


        def batch_norm(self, x, n_out, phase_train):
                beta = tf.Variable(tf.constant(0.0, shape=[n_out]),name='beta', trainable=True)
                gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),name='gamma', trainable=True)
                batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
                ema = tf.train.ExponentialMovingAverage(decay=0.5)
                def mean_var_with_update():
                        ema_apply_op = ema.apply([batch_mean, batch_var])
                        with tf.control_dependencies([ema_apply_op]):
                                return tf.identity(batch_mean), tf.identity(batch_var)
                mean, var = tf.cond(phase_train,mean_var_with_update,lambda: (ema.average(batch_mean), ema.average(batch_var)))
                normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
                return normed

	def build(self, rgb, value, keep_prob):

		# conv1_1
		with tf.name_scope('teacher_conv1_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([5,5,3,192], dtype = tf.float32, stddev = 1e-2), name='weights', trainable = value)
                        conv = tf.nn.conv2d(rgb, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(1.0, shape = [192], dtype = tf.float32), name='biases', trainable = value)
			out = tf.nn.bias_add(conv, biases)
                        
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
                        
			self.conv1_1 = tf.nn.relu(batch_norm, name=scope)
		        self.parameters += [kernel, biases]
                        
		with tf.name_scope('teacher_conv2_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([1,1,192,160]), name='weights', trainable = value)
			conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(1.0, shape = [160], dtype = tf.float32), name='biases', trainable = value)
			out = tf.nn.bias_add(conv, biases)
                        
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
                        
			self.conv2_1 = tf.nn.relu(batch_norm, name=scope)
			self.parameters += [kernel, biases]

		with tf.name_scope('teacher_conv3_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([1,1,160,96]), name='weights', trainable = value)
			conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1,1], padding='SAME')
			biases = tf.Variable(tf.constant(1.0, shape = [96], dtype = tf.float32), name='biases', trainable = value)
			out = tf.nn.bias_add(conv, biases)
                        
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
                        
			self.conv3_1 = tf.nn.relu(batch_norm, name=scope)
		        self.parameters += [kernel, biases]
                self.pool1 = tf.nn.max_pool(self.conv3_1,
                        ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        name='pool1')
                        
                self.dropout1 = tf.nn.dropout(self.pool1, keep_prob=0.5)
		with tf.name_scope('teacher_conv4_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([5, 5, 96,192], dtype = tf.float32, stddev = 1e-2), name='weights', trainable = value)
			conv = tf.nn.conv2d(self.dropout1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(1.0, shape = [192], dtype = tf.float32), name='biases', trainable = value)
			out = tf.nn.bias_add(conv, biases)
                        
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
                        
			self.conv4_1 = tf.nn.relu(batch_norm, name=scope)
			self.parameters += [kernel, biases]

		with tf.name_scope('teacher_conv5_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([1, 1, 192,192], dtype = tf.float32, stddev = 1e-2), name='weights', trainable = value)
			conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(1.0, shape = [192], dtype = tf.float32), name='biases', trainable = value)
			out = tf.nn.bias_add(conv, biases)
                        
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
                        
			self.conv5_1 = tf.nn.relu(batch_norm, name=scope)
			self.parameters += [kernel, biases]

		with tf.name_scope('teacher_conv6_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([1, 1, 192,192], dtype = tf.float32, stddev = 1e-2), name='weights', trainable = value)
			conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(1.0, shape = [192], dtype = tf.float32), name='biases', trainable = value)
			out = tf.nn.bias_add(conv, biases)
                        
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
                        
			self.conv6_1 = tf.nn.relu(batch_norm, name=scope)
			self.parameters += [kernel, biases]

                self.pool2 = tf.nn.avg_pool(self.conv6_1,
                        ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        name='pool2')
                self.dropout2 = tf.nn.dropout(self.pool2, keep_prob=0.5)
		
                with tf.name_scope('teacher_conv7_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 192,192], dtype = tf.float32, stddev = 1e-2), name='weights', trainable = value)
			conv = tf.nn.conv2d(self.dropout2, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(1.0, shape = [192], dtype = tf.float32), name='biases', trainable = value)
			out = tf.nn.bias_add(conv, biases)
                        
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
                        
			self.conv7_1 = tf.nn.relu(batch_norm, name=scope)
		with tf.name_scope('teacher_conv8_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([1, 1, 192,192], dtype = tf.float32, stddev = 1e-2), name='weights', trainable = value)
			conv = tf.nn.conv2d(self.conv7_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(1.0, shape = [192], dtype = tf.float32), name='biases', trainable = value)
			out = tf.nn.bias_add(conv, biases)
                        
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
                        
			self.conv8_1 = tf.nn.relu(batch_norm, name=scope)
		with tf.name_scope('teacher_conv9_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([1, 1, 192,NUM_CLASSES], dtype = tf.float32, stddev = 1e-2), name='weights', trainable =value)
			conv = tf.nn.conv2d(self.conv8_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(1.0, shape = [NUM_CLASSES], dtype = tf.float32), name='biases', trainable = value)
			out = tf.nn.bias_add(conv, biases)
                        
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
                        
			self.conv9_1 = tf.nn.relu(batch_norm, name=scope)

                
                self.pool3 = tf.nn.avg_pool(self.conv9_1,
                        ksize=[1, 8, 8, 1],
                        strides=[1, 1, 1,1],
                        padding='VALID',
                        name='pool3')
                
                
                shape = int(np.prod(self.pool3.get_shape()[1:]))
                self.pool3 = tf.reshape(self.pool3, [-1, shape])

                epsilon = tf.random_normal([1,10], mean=0.0, stddev = 0.8, dtype = tf.float32)

                ones_vector = tf.ones([1,10])
                noise = tf.add(ones_vector,epsilon)
                self.pool3 =  tf.multiply(self.pool3, noise)
                
                #logits_temp = tf.divide(self.dropout3, tf.constant(TEMP_SOFTMAX))
                return  self.pool3
        
        def loss(self, labels):
                labels = tf.to_int64(labels)
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=self.pool3, name='xentropy')
                return tf.reduce_mean(cross_entropy, name='xentropy')

	def training(self, loss, learning_rate, global_step):
		tf.summary.scalar('loss', loss)

		optimizer = tf.train.AdamOptimizer(learning_rate)
		
		train_op = optimizer.minimize(loss, global_step=global_step)

                return train_op
