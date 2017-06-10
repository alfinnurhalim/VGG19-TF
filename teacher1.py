import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import numpy as np
NUM_CLASSES = 10
beta = 0.004
TEMP_SOFTMAX = 3.0
VGG_MEAN = [103.939, 116.779, 123.68]

class Teacher(object):

	def __init__(self, trainable=True, dropout=0.5):
		self.trainable = trainable
		self.dropout = dropout
		self.parameters = []

	def build(self, rgb, value,train_mode=None):

		"""
		:param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
		:param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
		"""
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
                """
		# conv1_1
		with tf.name_scope('teacher_conv1_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([5,5,3,64], dtype = tf.float32, stddev = 5e-2), name='weights', trainable = value)
                        conv = tf.nn.conv2d(rgb, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape = [64], dtype = tf.float32), name='biases', trainable = value)
			out = tf.nn.bias_add(conv, biases)
                        """
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
                        """
			self.conv1_1 = tf.nn.relu(out, name=scope)
                        
                        #tf.nn.dropout(self.conv1_1, keep_prob)
			self.parameters += [kernel, biases]
                
                self.pool1 = tf.nn.max_pool(self.conv1_1,
                        ksize=[1,3, 3, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        name='pool1')

                norm1 = tf.nn.lrn(self.pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')

		with tf.name_scope('teacher_conv2_1') as scope:
                        kernel = tf.Variable(tf.truncated_normal([5,5,64,64],dtype= tf.float32, stddev = 5e-2), name='weights', trainable = value)
		        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
		        biases = tf.Variable(tf.constant(0.1, shape = [64], dtype = tf.float32), name='biases', trainable = value)
			out = tf.nn.bias_add(conv, biases)
                        """
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
                        """
			self.conv2_1 = tf.nn.relu(out, name=scope)
                        #tf.nn.dropout(self.conv2_1, keep_prob)
			self.parameters += [kernel, biases]
                norm2 = tf.nn.lrn(self.conv2_1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                                name='norm2')

                self.pool2 = tf.nn.max_pool(norm2,
                        ksize=[1,3, 3, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        name='pool1')

		# fc1
		with tf.name_scope('teacher_fc1') as scope:
			shape = int(np.prod(self.pool2.get_shape()[1:]))
			fc1w = tf.Variable(tf.truncated_normal([shape,384], dtype = tf.float32, stddev = 0.04), name='weights', trainable = value)
			fc1b = tf.Variable(tf.constant(0.1, shape = [384], dtype = tf.float32), name='biases', trainable = value)
			pool2_flat = tf.reshape(self.pool2, [-1, shape])
			fc1l = tf.nn.bias_add(tf.matmul(pool2_flat, fc1w), fc1b)

			self.fc1 = tf.nn.relu(fc1l)
                        
                        #self.fc1 = tf.nn.dropout(self.fc1, keep_prob)
			self.parameters += [fc1w, fc1b]


		with tf.name_scope('teacher_fc2') as scope:
			fc2w = tf.Variable(tf.truncated_normal([384,192],
														 dtype=tf.float32, stddev=0.04), name='weights', trainable = value)
			fc2b = tf.Variable(tf.constant(0.1, shape=[192], dtype=tf.float32),
								  name='biases', trainable = value)
			self.fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
                        self.fc2l = tf.nn.relu(self.fc2l)
                        #self.fc2l = tf.nn.dropout(self.fc2l, keep_prob)
			self.parameters += [fc2w, fc2b]  

                
		# fc2
		with tf.name_scope('teacher_fc3') as scope:
			fc3w = tf.Variable(tf.truncated_normal([192, NUM_CLASSES],
														 dtype=tf.float32, stddev=1/192.0), name='weights', trainable = value)
			fc3b = tf.Variable(tf.constant(0.0, shape=[NUM_CLASSES], dtype=tf.float32),
								  name='biases', trainable = value)
			self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2l, fc3w), fc3b)
			self.parameters += [fc3w, fc3b]  
                logits_temp = tf.divide(self.fc3l, tf.constant(TEMP_SOFTMAX))

                
                return self.pool2, logits_temp 
                #return self.conv5_1, self.conv5_3, self.fc3l, tf.nn.softmax(logits_temp)
        
        def variables_for_l2(self):
            
                variables_for_l2 = []
                
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "teacher_fc1/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "teacher_fc2/weights:0"][0])
            
            

                return variables_for_l2


	def loss(self, labels):
		labels = tf.to_int64(labels)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
			logits=self.fc3l, name='xentropy')
                var_list = self.variables_for_l2()
                
                l2_loss= beta*tf.nn.l2_loss(var_list[0]) + beta*tf.nn.l2_loss(var_list[1])
                
		return tf.reduce_mean(tf.add(cross_entropy,l2_loss), name='xentropy')


	def training(self, loss, learning_rate, global_step):
		tf.summary.scalar('loss', loss)

                ### Adding Momentum of 0.9
		#optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_locking=False,name='Momentum',
                  #          use_nesterov=True)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		train_op = optimizer.minimize(loss, global_step=global_step)

                return train_op
