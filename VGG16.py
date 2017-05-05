import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import numpy as np
NUM_CLASSES = 102
beta = 0.001
TEMP_SOFTMAX = 5.0
VGG_MEAN = [103.939, 116.779, 123.68]

class VGG16(object):

	def __init__(self, trainable=True, dropout=0.5):
		self.trainable = trainable
		self.dropout = dropout
                self.data_dict = np.load("vgg16.npy").item()
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
			kernel = tf.Variable(self.data_dict["conv1_1"][0], name='weights', trainable = True)
                        conv = tf.nn.conv2d(rgb, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(self.data_dict["conv1_1"][1], name='biases', trainable = True)
			#out = tf.nn.bias_add(conv, biases)
                        #mean, var = tf.nn.moments(out, axes=[0])
                        #batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))

			self.conv1_1 = tf.nn.relu(conv, name=scope)
			self.parameters += [kernel, biases]

			
		# conv1_2
		with tf.name_scope('conv1_2') as scope:
			kernel = tf.Variable(self.data_dict["conv1_2"][0], name='weights', trainable = True)
			conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(self.data_dict["conv1_2"][1], name='biases', trainable = True)
			#out = tf.nn.bias_add(conv, biases)
                        #mean, var = tf.nn.moments(out, axes=[0])
                        #batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv1_2 = tf.nn.relu(conv, name=scope)
                        #tf.nn.dropout(self.conv1_2, keep_prob=0.7)

			self.parameters += [kernel, biases]

		self.pool1 = tf.nn.max_pool(self.conv1_2,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
    									name='pool1')
                        
		with tf.name_scope('conv2_1') as scope:
			kernel = tf.Variable(self.data_dict["conv2_1"][0], name='weights', trainable = True)
			conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(self.data_dict["conv2_1"][1], name='biases', trainable = True)
			#out = tf.nn.bias_add(conv, biases)
                        #mean, var = tf.nn.moments(out, axes=[0])
                        #batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv2_1 = tf.nn.relu(conv, name=scope)
                        #tf.nn.dropout(self.conv2_1, keep_prob=0.6)
			self.parameters += [kernel, biases]

		with tf.name_scope('conv2_2') as scope:
			kernel = tf.Variable(self.data_dict["conv2_2"][0], name='weights', trainable = True)
			conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(self.data_dict["conv2_2"][1], name='biases', trainable = True)
			#out = tf.nn.bias_add(conv, biases)
                        #mean, var = tf.nn.moments(out, axes=[0])
                        #batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv2_2 = tf.nn.relu(conv, name=scope)
                        #tf.nn.dropout(self.conv2_2, keep_prob=0.6)
			self.parameters += [kernel, biases]

		self.pool2 = tf.nn.max_pool(self.conv2_2,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool2')

		with tf.name_scope('conv3_1') as scope:
			kernel = tf.Variable(self.data_dict["conv3_1"][0], name='weights', trainable = True)
			conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(self.data_dict["conv3_1"][1], name='biases', trainable = True)
			#out = tf.nn.bias_add(conv, biases)
                        #mean, var = tf.nn.moments(out, axes=[0])
                        #batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv3_1 = tf.nn.relu(conv, name=scope)
                        #tf.nn.dropout(self.conv3_1, keep_prob=0.6)
			self.parameters += [kernel, biases]

		with tf.name_scope('conv3_2') as scope:
			kernel = tf.Variable(self.data_dict["conv3_2"][0], name='weights', trainable = True)
			conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(self.data_dict["conv3_2"][1], name='biases', trainable = True)
			#out = tf.nn.bias_add(conv, biases)
                        #mean, var = tf.nn.moments(out, axes=[0])
                        #batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv3_2 = tf.nn.relu(conv, name=scope)
                        #tf.nn.dropout(self.conv3_2, keep_prob=0.6)
			self.parameters += [kernel, biases]

		with tf.name_scope('conv3_3') as scope:
			kernel = tf.Variable(self.data_dict["conv3_3"][0], name='weights', trainable = True)
			conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(self.data_dict["conv3_3"][1], name='biases', trainable = True)
			#out = tf.nn.bias_add(conv, biases)
                        #mean, var = tf.nn.moments(out, axes=[0])
                        #batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv3_3 = tf.nn.relu(conv, name=scope)
                        #tf.nn.dropout(self.conv3_3, keep_prob=0.6)
			self.parameters += [kernel, biases]

		self.pool3 = tf.nn.max_pool(self.conv3_3,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool3')

		# conv4_1
		with tf.name_scope('conv4_1') as scope:
			kernel = tf.Variable(self.data_dict["conv4_1"][0], name='weights', trainable = True)
			conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(self.data_dict["conv4_1"][1], name='biases', trainable = True)
			#out = tf.nn.bias_add(conv, biases)
                        #mean, var = tf.nn.moments(out, axes=[0])
                        #batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv4_1 = tf.nn.relu(conv, name=scope)
                        #tf.nn.dropout(self.conv4_1, keep_prob=0.6)
			self.parameters += [kernel, biases]

		# conv4_2
		with tf.name_scope('conv4_2') as scope:
			kernel = tf.Variable(self.data_dict["conv4_2"][0], name='weights', trainable = True)
			conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(self.data_dict["conv4_2"][1], name='biases', trainable = True)
			#out = tf.nn.bias_add(conv, biases)
                        #mean, var = tf.nn.moments(out, axes=[0])
                        #batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv4_2 = tf.nn.relu(conv, name=scope)
                        #tf.nn.dropout(self.conv4_2, keep_prob=0.6)
			self.parameters += [kernel, biases]

		# conv4_3
		with tf.name_scope('conv4_3') as scope:
			kernel = tf.Variable(self.data_dict["conv4_3"][0], name='weights', trainable = True)
			conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(self.data_dict["conv4_3"][1], name='biases', trainable = True)
			#out = tf.nn.bias_add(conv, biases)
                        #mean, var = tf.nn.moments(out, axes=[0])
                        #batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv4_3 = tf.nn.relu(conv, name=scope)
                        #tf.nn.dropout(self.conv4_3, keep_prob=0.6)
			self.parameters += [kernel, biases]

		# pool4
		self.pool4 = tf.nn.max_pool(self.conv4_3,
							   ksize=[1, 2, 2, 1],
							   strides=[1, 2, 2, 1],
							   padding='SAME',
							   name='pool4')


		# conv5_1
		with tf.name_scope('conv5_1') as scope:
			kernel = tf.Variable(self.data_dict["conv5_1"][0], name='weights', trainable = True)
			conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(self.data_dict["conv5_1"][1], name='biases', trainable = True)
			#out = tf.nn.bias_add(conv, biases)
                        #mean, var = tf.nn.moments(out, axes=[0])
                        #batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv5_1 = tf.nn.relu(conv, name=scope)
                        #tf.nn.dropout(self.conv5_1, keep_prob=0.6)
			self.parameters += [kernel, biases]

		# conv5_2
		with tf.name_scope('conv5_2') as scope:
			kernel = tf.Variable(self.data_dict["conv5_2"][0], name='weights', trainable = True)
			conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(self.data_dict["conv5_2"][1], name='biases', trainable = True)
			#out = tf.nn.bias_add(conv, biases)
                        #mean, var = tf.nn.moments(out, axes=[0])
                        #batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv5_2 = tf.nn.relu(conv, name=scope)
                        #tf.nn.dropout(self.conv5_2, keep_prob=0.6)
			self.parameters += [kernel, biases]

		# conv5_3
		with tf.name_scope('conv5_3') as scope:
			kernel = tf.Variable(self.data_dict["conv5_3"][0], name='weights', trainable = True)
			conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(self.data_dict["conv5_3"][1], name='biases', trainable = True)
			#out = tf.nn.bias_add(conv, biases)
                        #mean, var = tf.nn.moments(out, axes=[0])
                        #batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
			self.conv5_3 = tf.nn.relu(conv, name=scope)
                        tf.nn.dropout(self.conv5_3, keep_prob=0.6)
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
			fc1w = tf.Variable(self.data_dict["fc6"][0], name='weights', trainable = True)
			fc1b = tf.Variable(self.data_dict["fc6"][1], name='biases', trainable = True)
			pool5_flat = tf.reshape(self.pool5, [-1, shape])
			fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
			self.fc1 = tf.nn.relu(fc1l)
                        
                        self.fc1 = tf.nn.dropout(self.fc1, 0.5)
			self.parameters += [fc1w, fc1b]


		# fc2
		with tf.name_scope('fc2') as scope:
			fc2w = tf.Variable(self.data_dict["fc7"][0], name='weights', trainable = True)
			fc2b = tf.Variable(self.data_dict["fc7"][1], name='biases', trainable = True)
			fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
			self.fc2 = tf.nn.relu(fc2l)
                        
                        self.fc2 = tf.nn.dropout(self.fc2, 0.5)
			self.parameters += [fc2w, fc2b]

		# fc3
		with tf.name_scope('fc3') as scope:
			fc3w = tf.Variable(tf.truncated_normal([4096, NUM_CLASSES],
														 dtype=tf.float32, stddev=1e-2), name='weights', trainable = True)
			fc3b = tf.Variable(tf.constant(1.0, shape=[NUM_CLASSES], dtype=tf.float32),
								  name='biases', trainable = True)
			self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
			self.parameters += [fc3w, fc3b]  
                logits_temp = tf.divide(self.fc3l, tf.constant(TEMP_SOFTMAX))

                return self.conv5_1, self.conv5_3, self.fc3l, tf.nn.softmax(logits_temp)
        
        def variables_for_l2(self):
            
                variables_for_l2 = []
                variables_for_l2.append([var for var in tf.global_variables() if var.op.name=="conv1_1/weights"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "conv1_2/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "conv2_1/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "conv2_2/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "conv3_1/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "conv3_2/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "conv3_3/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "conv4_1/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "conv4_2/weights:0"][0])
                variables_for_l2.append ([v for v in tf.global_variables() if v.name == "conv4_3/weights:0"][0])
                variables_for_l2.append ([v for v in tf.global_variables() if v.name == "conv5_1/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "conv5_2/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "conv5_3/weights:0"][0])        
                
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "fc1/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "fc2/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "fc3/weights:0"][0])
            
            

                return variables_for_l2


        def train_last_layer_variables(self, train_last_layer_variables):

                
                train_last_layer_variables.append([v for v in tf.global_variables() if v.name == "fc3/weights:0"][0])
                train_last_layer_variables.append([v for v in tf.global_variables() if v.name == "fc3/biases:0"][0])
                return train_last_layer_variables

	def loss(self, labels):
		labels = tf.to_int64(labels)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
			logits=self.fc3l, name='xentropy')
                var_list = self.variables_for_l2()
                """
                l2_loss= beta*tf.nn.l2_loss(var_list[0]) + beta*tf.nn.l2_loss(var_list[1]) + beta*tf.nn.l2_loss(var_list[2]) + beta*tf.nn.l2_loss(var_list[3]) + beta*tf.nn.l2_loss(var_list[4]) + beta*tf.nn.l2_loss(var_list[4]) + beta*tf.nn.l2_loss(var_list[5]) + beta*tf.nn.l2_loss(var_list[6])+beta*tf.nn.l2_loss(var_list[7])+beta*tf.nn.l2_loss(var_list[8]) + beta*tf.nn.l2_loss(var_list[9]) + beta*tf.nn.l2_loss(var_list[10]) + beta*tf.nn.l2_loss(var_list[11]) + beta*tf.nn.l2_loss(var_list[12]) + beta*tf.nn.l2_loss(var_list[13])+beta*tf.nn.l2_loss(var_list[14])
                """
		return tf.reduce_mean(cross_entropy, name='xentropy_mean')


	def training(self, loss, learning_rate, learning_rate_pretrained, train_last_layer_variables, variables_to_restore):
		tf.summary.scalar('loss', loss)

                ### Adding Momentum of 0.9
		optimizer1 = tf.train.AdamOptimizer(learning_rate_pretrained)
		optimizer2 = tf.train.AdamOptimizer(learning_rate)
		
		self.global_step = tf.Variable(0, name='global_step', trainable=False)
		train_op1 = optimizer1.minimize(loss, global_step=self.global_step, var_list = variables_to_restore)

		train_op2 = optimizer2.minimize(loss, global_step=self.global_step, var_list = train_last_layer_variables)
		#train_op = optimizer.minimize(loss)

		return tf.group(train_op1, train_op2)
