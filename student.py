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
			kernel = tf.Variable(tf.truncated_normal([3,3,3,32], dtype = tf.float32, stddev = 1e-2), name='weights', trainable = True)
                        conv = tf.nn.conv2d(rgb, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape = [32], dtype = tf.float32), name='biases', trainable = True)
			out = tf.nn.bias_add(conv, biases)
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))

			self.conv1_1 = tf.nn.relu(batch_norm, name=scope)
			self.parameters += [kernel, biases]


		with tf.name_scope('student_conv2_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,32,32]), name='weights', trainable = True)
			conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape = [32], dtype = tf.float32), name='biases', trainable = True)
			out = tf.nn.bias_add(conv, biases)
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv2_1 = tf.nn.relu(batch_norm, name=scope)
                        #tf.nn.dropout(self.conv2_1, keep_prob=0.6)
			self.parameters += [kernel, biases]

		with tf.name_scope('student_conv3_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 32,32], dtype = tf.float32, stddev = 1e-2), name='weights', trainable = True)
			conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape = [32], dtype = tf.float32), name='biases', trainable = True)
			out = tf.nn.bias_add(conv, biases)
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv3_1 = tf.nn.relu(batch_norm, name=scope)
                        #tf.nn.dropout(self.conv3_1, keep_prob=0.6)
			self.parameters += [kernel, biases]


		with tf.name_scope('student_conv4_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 32,48], dtype = tf.float32, stddev = 1e-2), name='weights', trainable = True)
			conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape = [48], dtype = tf.float32), name='biases', trainable = True)
			out = tf.nn.bias_add(conv, biases)
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv4_1 = tf.nn.relu(batch_norm, name=scope)
                        #tf.nn.dropout(self.conv3_1, keep_prob=0.6)
			self.parameters += [kernel, biases]


		with tf.name_scope('student_conv5_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 48,48], dtype = tf.float32, stddev = 1e-2), name='weights', trainable = True)
			conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape = [48], dtype = tf.float32), name='biases', trainable = True)
			out = tf.nn.bias_add(conv, biases)
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv5_1 = tf.nn.relu(batch_norm, name=scope)
                        #tf.nn.dropout(self.conv3_1, keep_prob=0.6)
			self.parameters += [kernel, biases]

                self.pool1 = tf.nn.max_pool(self.conv5_1,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME',
                                name='pool1')

		with tf.name_scope('student_conv6_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 48,80], dtype = tf.float32, stddev = 1e-2), name='weights', trainable = True)
			conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape = [80], dtype = tf.float32), name='biases', trainable = True)
			out = tf.nn.bias_add(conv, biases)
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv6_1 = tf.nn.relu(batch_norm, name=scope)
                        #tf.nn.dropout(self.conv3_1, keep_prob=0.6)
			self.parameters += [kernel, biases]

		with tf.name_scope('student_conv7_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 80,80], dtype = tf.float32, stddev = 1e-2), name='weights', trainable = True)
			conv = tf.nn.conv2d(self.conv6_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape = [80], dtype = tf.float32), name='biases', trainable = True)
			out = tf.nn.bias_add(conv, biases)
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv7_1 = tf.nn.relu(batch_norm, name=scope)
                        #tf.nn.dropout(self.conv3_1, keep_prob=0.6)
			self.parameters += [kernel, biases]


		with tf.name_scope('student_conv8_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 80,80], dtype = tf.float32, stddev = 1e-2), name='weights', trainable = True)
			conv = tf.nn.conv2d(self.conv7_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape = [80], dtype = tf.float32), name='biases', trainable = True)
			out = tf.nn.bias_add(conv, biases)
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv8_1 = tf.nn.relu(batch_norm, name=scope)
                        #tf.nn.dropout(self.conv3_1, keep_prob=0.6)
			self.parameters += [kernel, biases]


		with tf.name_scope('student_conv9_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 80,80], dtype = tf.float32, stddev = 1e-2), name='weights', trainable = True)
			conv = tf.nn.conv2d(self.conv8_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape = [80], dtype = tf.float32), name='biases', trainable = True)
			out = tf.nn.bias_add(conv, biases)
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv9_1 = tf.nn.relu(batch_norm, name=scope)
                        #tf.nn.dropout(self.conv3_1, keep_prob=0.6)
			self.parameters += [kernel, biases]


		with tf.name_scope('student_conv10_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 80,80], dtype = tf.float32, stddev = 1e-2), name='weights', trainable = True)
			conv = tf.nn.conv2d(self.conv9_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape = [80], dtype = tf.float32), name='biases', trainable = True)
			out = tf.nn.bias_add(conv, biases)
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv10_1 = tf.nn.relu(batch_norm, name=scope)
                        #tf.nn.dropout(self.conv3_1, keep_prob=0.6)
			self.parameters += [kernel, biases]

		with tf.name_scope('student_conv11_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 80,80], dtype = tf.float32, stddev = 1e-2), name='weights', trainable = True)
			conv = tf.nn.conv2d(self.conv10_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape = [80], dtype = tf.float32), name='biases', trainable = True)
			out = tf.nn.bias_add(conv, biases)
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv11_1 = tf.nn.relu(batch_norm, name=scope)
                        #tf.nn.dropout(self.conv3_1, keep_prob=0.6)
			self.parameters += [kernel, biases]

                self.pool2 = tf.nn.max_pool(self.conv11_1,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME',
                                name='pool2')


		with tf.name_scope('student_conv12_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 80,128], dtype = tf.float32, stddev = 1e-2), name='weights', trainable = True)
			conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape = [128], dtype = tf.float32), name='biases', trainable = True)
			out = tf.nn.bias_add(conv, biases)
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv12_1 = tf.nn.relu(batch_norm, name=scope)
                        #tf.nn.dropout(self.conv3_1, keep_prob=0.6)
			self.parameters += [kernel, biases]

		with tf.name_scope('student_conv13_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 128,128], dtype = tf.float32, stddev = 1e-2), name='weights', trainable = True)
			conv = tf.nn.conv2d(self.conv12_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape = [128], dtype = tf.float32), name='biases', trainable = True)
			out = tf.nn.bias_add(conv, biases)
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv13_1 = tf.nn.relu(batch_norm, name=scope)
                        #tf.nn.dropout(self.conv3_1, keep_prob=0.6)
			self.parameters += [kernel, biases]

		with tf.name_scope('student_conv14_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 128,128], dtype = tf.float32, stddev = 1e-2), name='weights', trainable = True)
			conv = tf.nn.conv2d(self.conv13_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape = [128], dtype = tf.float32), name='biases', trainable = True)
			out = tf.nn.bias_add(conv, biases)
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv14_1 = tf.nn.relu(batch_norm, name=scope)
                        #tf.nn.dropout(self.conv3_1, keep_prob=0.6)
			self.parameters += [kernel, biases]


		with tf.name_scope('student_conv15_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 128,128], dtype = tf.float32, stddev = 1e-2), name='weights', trainable = True)
			conv = tf.nn.conv2d(self.conv14_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape = [128], dtype = tf.float32), name='biases', trainable = True)
			out = tf.nn.bias_add(conv, biases)
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv15_1 = tf.nn.relu(batch_norm, name=scope)
                        #tf.nn.dropout(self.conv3_1, keep_prob=0.6)
			self.parameters += [kernel, biases]


		with tf.name_scope('student_conv16_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 128,128], dtype = tf.float32, stddev = 1e-2), name='weights', trainable = True)
			conv = tf.nn.conv2d(self.conv15_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape = [128], dtype = tf.float32), name='biases', trainable = True)
			out = tf.nn.bias_add(conv, biases)
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv16_1 = tf.nn.relu(batch_norm, name=scope)
                        #tf.nn.dropout(self.conv3_1, keep_prob=0.6)
			self.parameters += [kernel, biases]


		with tf.name_scope('student_conv17_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 128,128], dtype = tf.float32, stddev = 1e-2), name='weights', trainable = True)
			conv = tf.nn.conv2d(self.conv16_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape = [128], dtype = tf.float32), name='biases', trainable = True)
			out = tf.nn.bias_add(conv, biases)
                        mean, var = tf.nn.moments(out, axes=[0])
                        batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv17_1 = tf.nn.relu(batch_norm, name=scope)
                        #tf.nn.dropout(self.conv3_1, keep_prob=0.6)
			self.parameters += [kernel, biases]

                self.pool3 = tf.nn.max_pool(self.conv17_1,
                                ksize=[1, 8, 8, 1],
                                strides=[1, 1, 1, 1],
                                padding='SAME',
                                name='pool3')
		# fc1
		with tf.name_scope('student_fc1') as scope:
			shape = int(np.prod(self.pool3.get_shape()[1:]))
			fc1w = tf.Variable(tf.truncated_normal([shape,500]), name='weights', trainable = True)
			fc1b = tf.Variable(tf.constant(1.0, shape = [500], dtype = tf.float32), name='biases', trainable = True)
			pool3_flat = tf.reshape(self.pool3, [-1, shape])
			fc1 = tf.nn.bias_add(tf.matmul(pool3_flat, fc1w), fc1b)
			self.fc1 = tf.nn.relu(fc1)
                        
                        self.fc1 = tf.nn.dropout(self.fc1, 0.5)
			self.parameters += [fc1w, fc1b]


		# fc2
		with tf.name_scope('student_fc2') as scope:
			fc2w = tf.Variable(tf.truncated_normal([500, NUM_CLASSES],
														 dtype=tf.float32, stddev=1e-2), name='weights', trainable = True)
			fc2b = tf.Variable(tf.constant(1.0, shape=[NUM_CLASSES], dtype=tf.float32),
								  name='biases', trainable = True)
			self.fc2 = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
			self.parameters += [fc2w, fc2b]  
                logits_temp = tf.divide(self.fc2, tf.constant(TEMP_SOFTMAX))

                #return self.conv5_1, self.conv5_3, self.fc3l, tf.nn.softmax(logits_temp)
                return self.pool2, logits_temp 
        def variables_for_l2(self):
            
                variables_for_l2 = []
                variables_for_l2.append([var for var in tf.global_variables() if var.op.name=="student_conv1_1/weights"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "student_conv2_1/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "student_conv3_1/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "student_conv4_1/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "student_conv5_1/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "student_conv6_1/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "student_conv7_1/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "student_conv8_1/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "student_conv9_1/weights:0"][0])
                variables_for_l2.append ([v for v in tf.global_variables() if v.name == "student_conv10_1/weights:0"][0])
                variables_for_l2.append ([v for v in tf.global_variables() if v.name == "student_conv11_1/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "student_conv12_1/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "student_conv13_1/weights:0"][0])        
                
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "student_conv14_1/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "student_conv15_1/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "student_conv16_1/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "student_conv17_1/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "student_fc1/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "student_fc2/weights:0"][0])

                return variables_for_l2


	def loss(self, labels):
		labels = tf.to_int64(labels)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=self.fc2, name='xentropy')
                var_list = self.variables_for_l2()
                
                l2_loss= beta*tf.nn.l2_loss(var_list[0]) + beta*tf.nn.l2_loss(var_list[1]) + beta*tf.nn.l2_loss(var_list[2]) + beta*tf.nn.l2_loss(var_list[3]) + beta*tf.nn.l2_loss(var_list[4]) + beta*tf.nn.l2_loss(var_list[4]) + beta*tf.nn.l2_loss(var_list[5]) + beta*tf.nn.l2_loss(var_list[6])+beta*tf.nn.l2_loss(var_list[7])+beta*tf.nn.l2_loss(var_list[8]) + beta*tf.nn.l2_loss(var_list[9]) + beta*tf.nn.l2_loss(var_list[10]) + beta*tf.nn.l2_loss(var_list[11]) + beta*tf.nn.l2_loss(var_list[12]) + beta*tf.nn.l2_loss(var_list[13])+beta*tf.nn.l2_loss(var_list[14])+beta*tf.nn.l2_loss(var_list[15])+ beta*tf.nn.l2_loss(var_list[16])+beta*tf.nn.l2_loss(var_list[17])+ beta*tf.nn.l2_loss(var_list[18]) 
                
		return tf.reduce_mean(cross_entropy, name='xentropy_mean') + l2_loss


	def training(self, loss, learning_rate):
		tf.summary.scalar('loss', loss)

                ### Adding Momentum of 0.9
		#optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum = 0.0) 
		optimizer = tf.train.AdamOptimizer(learning_rate)
		
		self.global_step = tf.Variable(0, name='global_step', trainable=False)
		train_op = optimizer.minimize(loss, global_step=self.global_step)

                return train_op
