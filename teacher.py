import tensorflow as tf 
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import numpy as np
beta = 0.001
TEMP_SOFTMAX = 1.0

##Teacher class consists of 6 convolutional layers and 3 fully connected layers trained from scratch

class Teacher(object):

	def __init__(self, trainable=True, dropout=0.5):
                #trainable: boolean value set to True if the layers are made trainable else set to False
		self.trainable = trainable
                ## parameters is a list which stores weights and biases of all the layers of the teacher model
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

	def build(self, rgb, keep_prob, num_classes):

                        #rgb: Input to the teacher network
                        #num_classes: num of output classes in the dataset.
		
		with tf.name_scope('teacher_conv1_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,3,128], dtype = tf.float32, stddev = 1e-2), name='weights', trainable = self.trainable)
                        conv = tf.nn.conv2d(rgb, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(1.0, shape = [128], dtype = tf.float32), name='biases', trainable = self.trainable)
			out = tf.nn.bias_add(conv, biases)
                        
                        ## below commented lines can be uncommented if batch norm is required to add after convolution layer
                        #mean, var = tf.nn.moments(out, axes=[0])
                        #batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
                        
			self.conv1_1 = tf.nn.relu(out, name=scope)
                        #self.conv1_1 = self.batch_norm(self.conv1_1, 128, phase_train)
		        self.parameters += [kernel, biases]
                        
		with tf.name_scope('teacher_conv2_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,128,128]), name='weights', trainable = self.trainable)
			conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(1.0, shape = [128], dtype = tf.float32), name='biases', trainable =self.trainable)
			out = tf.nn.bias_add(conv, biases)
                        
                        #mean, var = tf.nn.moments(out, axes=[0])
                        #batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
                        
			self.conv2_1 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

                self.pool2 = tf.nn.max_pool(self.conv2_1,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        name='pool2')

                ## uncomment below line if batch norm needs to be added after pooling layer
                #self.pool2 = self.batch_norm(self.pool2, 128, phase_train)
		with tf.name_scope('teacher_conv3_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,128,256]), name='weights', trainable = self.trainable)
			conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(1.0, shape = [256], dtype = tf.float32), name='biases', trainable = self.trainable)
			out = tf.nn.bias_add(conv, biases)
                        
                        #mean, var = tf.nn.moments(out, axes=[0])
                        #batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
                        
			self.conv3_1 = tf.nn.relu(out, name=scope)
		        self.parameters += [kernel, biases]

                        #self.conv3_1 = self.batch_norm(self.conv3_1, 256, phase_train)
		with tf.name_scope('teacher_conv4_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 256,256], dtype = tf.float32, stddev = 1e-2), name='weights', trainable =self.trainable)
			conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(1.0, shape = [256], dtype = tf.float32), name='biases', trainable = self.trainable)
			out = tf.nn.bias_add(conv, biases)
                        
                        #mean, var = tf.nn.moments(out, axes=[0])
                        #batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
                        
			self.conv4_1 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

                self.pool3 = tf.nn.max_pool(self.conv4_1,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        name='pool3')

                ## uncomment below line if batch norm needs to be added after pooling layer
                #self.pool3 = self.batch_norm(self.pool3, 256, phase_train)
		with tf.name_scope('teacher_conv5_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 256,512], dtype = tf.float32, stddev = 1e-2), name='weights', trainable = self.trainable)
			conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(1.0, shape = [512], dtype = tf.float32), name='biases', trainable = self.trainable)
			out = tf.nn.bias_add(conv, biases)
                        
                        #mean, var = tf.nn.moments(out, axes=[0])
                        #batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
                        
			self.conv5_1 = tf.nn.relu(out, name=scope)
                        #self.conv5_1 = self.batch_norm(self.conv5_1, 512, phase_train)
			self.parameters += [kernel, biases]

		with tf.name_scope('teacher_conv6_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 512,512], dtype = tf.float32, stddev = 1e-2), name='weights', trainable =self.trainable)
			conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(1.0, shape = [512], dtype = tf.float32), name='biases', trainable = self.trainable)
			out = tf.nn.bias_add(conv, biases)
                        
                        #mean, var = tf.nn.moments(out, axes=[0])
                        #batch_norm = (out - mean) / tf.sqrt(var + tf.constant(1e-10))
                        
			self.conv6_1 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

                self.pool4 = tf.nn.max_pool(self.conv6_1,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        name='pool4')
                #self.pool4 = self.batch_norm(self.pool4, 512, phase_train)

		# fully connected layer 1
		with tf.name_scope('teacher_fc1') as scope:
			shape = int(np.prod(self.pool4.get_shape()[1:]))
			fc1w = tf.Variable(tf.truncated_normal([shape,1024]), name='weights', trainable = self.trainable)
			fc1b = tf.Variable(tf.constant(1.0, shape = [1024], dtype = tf.float32), name='biases', trainable = self.trainable)
			pool4_flat = tf.reshape(self.pool4, [-1, shape])
			fc1 = tf.nn.bias_add(tf.matmul(pool4_flat, fc1w), fc1b)
                        
                        #mean, var = tf.nn.moments(fc1, axes=[0])
                        #batch_norm = (fc1 - mean) / tf.sqrt(var + tf.constant(1e-10))
                        
			self.fc1 = tf.nn.relu(fc1)
			self.parameters += [fc1w, fc1b]
		
                ## fully connected layer 2
                with tf.name_scope('teacher_fc2') as scope:
			
			fc2w = tf.Variable(tf.truncated_normal([1024,1024]), name='weights', trainable = self.trainable)
			fc2b = tf.Variable(tf.constant(1.0, shape = [1024], dtype = tf.float32), name='biases', trainable = self.trainable)
			fc2 = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
                        
                        #mean, var = tf.nn.moments(fc2, axes=[0])
                        #batch_norm = (fc2 - mean) / tf.sqrt(var + tf.constant(1e-10))
                        
			self.fc2 = tf.nn.relu(fc2)
                        self.fc2 = tf.nn.dropout(self.fc2, keep_prob)
			self.parameters += [fc2w, fc2b]

		## fully connected layer 3
		with tf.name_scope('teacher_fc3') as scope:
			fc3w = tf.Variable(tf.truncated_normal([1024, num_classes], dtype=tf.float32, stddev=1e-2), name='weights', trainable = value)
			fc3b = tf.Variable(tf.constant(1.0, shape=[num_classes], dtype=tf.float32),
								  name='biases', trainable = value)
			self.fc3 = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
                        #mean, var = tf.nn.moments(self.fc3l, axes=[0])
                        #batch_norm = (self.fc3l - mean) / tf.sqrt(var + tf.constant(1e-10))
                        self.fc3 = tf.nn.relu(self.fc3)
			self.parameters += [fc3w, fc3b]  

                ### TEMP_SOFTMAX is set to 1 for hard logits and is set to different values such as 5, 10 for soft logits
                self.logits_temp = tf.divide(self.fc3, tf.constant(TEMP_SOFTMAX))
                self.softmax_output = tf.nn.softmax(self.logits_temp)
                return self
        
        def variables_for_l2(self):
            
                ### Theses are the weight variables which should be added to l2 regularizer
                variables_for_l2 = []
                variables_for_l2.append([var for var in tf.global_variables() if var.op.name=="teacher_conv1_1/weights"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "teacher_conv2_1/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "teacher_conv3_1/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "teacher_conv4_1/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "teacher_conv5_1/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "teacher_conv6_1/weights:0"][0])
                
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "teacher_fc1/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "teacher_fc2/weights:0"][0])
                variables_for_l2.append([v for v in tf.global_variables() if v.name == "teacher_fc3/weights:0"][0])
            
            

                return variables_for_l2


	def loss(self, labels):
		labels = tf.to_int64(labels)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
			logits=self.fc3, name='xentropy')
                
                var_list = self.variables_for_l2()
                
                l2_loss= beta*tf.nn.l2_loss(var_list[0]) + beta*tf.nn.l2_loss(var_list[1]) + beta*tf.nn.l2_loss(var_list[2]) + beta*tf.nn.l2_loss(var_list[3]) + beta*tf.nn.l2_loss(var_list[4]) + beta*tf.nn.l2_loss(var_list[4]) + beta*tf.nn.l2_loss(var_list[5])+beta*tf.nn.l2_loss(var_list[6]) + beta*tf.nn.l2_loss(var_list[7])+beta*tf.nn.l2_loss(var_list[8])
                
                ### l2_loss can be added to the return statement provided we want to add l2 regularizer to the cross entropy loss.
                ### l2 regularizer will avoid moving the weights to infinity:: read more on L2 regularization
                ### return tf.reduce_mean(cross_entropy, name='xentropy') + l2_loss
		return tf.reduce_mean(cross_entropy, name='xentropy')


	def training(self, loss, learning_rate, global_step):
		tf.summary.scalar('loss', loss)

		optimizer = tf.train.AdamOptimizer(learning_rate)
		

                ### Adam optimizer minimizes the loss calculated above in the loss function
		train_op = optimizer.minimize(loss, global_step=global_step)

                return train_op
