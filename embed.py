import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import math
import numpy as np
import pdb
NUM_CLASSES = 102

VGG_MEAN = [103.939, 116.779, 123.68]
EMBED_UNITS = 64

class Embed(object):

	def __init__(self, trainable=True, dropout=0.5):
		self.trainable = trainable
		self.dropout = dropout
		self.parameters = []

	def build(self, data_dict_teacher, data_dict_student, train_mode):
            if(train_mode == 'HT'):
                with tf.name_scope('mentor_embed_1'):
                        shape = int(np.prod(data_dict_teacher.pool2.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True, name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentor_embed')
                        mentor_conv3_flat = tf.reshape(data_dict_teacher.pool2, [-1, shape])

                        embed_mentor_1 = tf.nn.bias_add(tf.matmul(mentor_conv3_flat, weights), biases)

                
                with tf.name_scope('mentee_embed_1'):
                        shape = int(np.prod(data_dict_stduent.pool2.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentee_embed')

                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True, name = 'biases_mentee_embed')

                        mentee_conv1_flat = tf.reshape(data_dict_student.pool2, [-1, shape])

                        embed_mentee_1 = tf.nn.bias_add(tf.matmul(mentee_conv1_flat, weights), biases)
                loss_embed_1 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_1, embed_mentee_1))))
                
                return loss_embed_1

            elif(train_mode == 'ML'):
                
                with tf.name_scope('mentor_embed_3'):
                        shape = int(np.prod(data_dict_teacher.conv1_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), name = 'biases_mentor_embed', trainable = True)

                        mentor_conv3_flat =tf.reshape(data_dict_teacher.conv1_1, [-1, shape])
                        embed_mentor_3 = tf.nn.bias_add(tf.matmul(mentor_conv3_flat, weights), biases)

                with tf.name_scope('mentee_embed_3'):
                        shape = int(np.prod(data_dict_student.conv1_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentee_embed')

                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentee_conv3_flat = tf.reshape(data_dict_student.conv1_1, [-1, shape])
                        embed_mentee_3 = tf.nn.bias_add(tf.matmul(mentee_conv3_flat, weights), biases)
                        

                with tf.name_scope('mentor_embed_4'):
                        shape = int(np.prod(data_dict_teacher.conv2_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), name = 'biases_mentor_embed', trainable = True)

                        mentor_conv4_flat =tf.reshape(data_dict_teacher.conv2_1, [-1, shape])
                        embed_mentor_4 = tf.nn.bias_add(tf.matmul(mentor_conv4_flat, weights), biases)

                with tf.name_scope('mentee_embed_4'):
                        shape = int(np.prod(data_dict_student.conv2_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentee_embed')

                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentee_conv4_flat = tf.reshape(data_dict_student.conv2_1, [-1, shape])
                        embed_mentee_4 = tf.nn.bias_add(tf.matmul(mentee_conv4_flat, weights), biases)
                        

                with tf.name_scope('mentor_embed_5'):
                        shape = int(np.prod(data_dict_teacher.conv3_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), name = 'biases_mentor_embed', trainable = True)
                        mentor_conv5_flat =tf.reshape(data_dict_teacher.conv3_1, [-1, shape])
                        embed_mentor_5 = tf.nn.bias_add(tf.matmul(mentor_conv5_flat, weights), biases)

                with tf.name_scope('mentee_embed_5'):
                        shape = int(np.prod(data_dict_student.conv3_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentee_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentee_conv5_flat = tf.reshape(data_dict_student.conv3_1, [-1, shape])
                        embed_mentee_5 = tf.nn.bias_add(tf.matmul(mentee_conv5_flat, weights), biases)
                 
                with tf.name_scope('mentor_embed_6'):
                        shape = int(np.prod(data_dict_teacher.conv1_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentor_conv6_flat = tf.reshape(data_dict_teacher.conv1_1, [-1, shape])
                        embed_mentor_6 = tf.nn.bias_add(tf.matmul(mentor_conv6_flat, weights), biases)
                
                with tf.name_scope('mentee_embed_6'):
                        shape = int(np.prod(data_dict_student.conv3_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentee_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentee_conv6_flat = tf.reshape(data_dict_student.conv3_1, [-1, shape])
                        embed_mentee_6 = tf.nn.bias_add(tf.matmul(mentee_conv6_flat, weights), biases)
                with tf.name_scope('mentor_embed_7'):
                        shape = int(np.prod(data_dict_teacher.conv2_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentor_conv7_flat = tf.reshape(data_dict_teacher.conv2_1, [-1, shape])
                        embed_mentor_7 = tf.nn.bias_add(tf.matmul(mentor_conv7_flat, weights), biases)
                
                with tf.name_scope('mentee_embed_7'):
                        shape = int(np.prod(data_dict_student.conv5_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentee_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentee_conv7_flat = tf.reshape(data_dict_student.conv5_1, [-1, shape])
                        embed_mentee_7 = tf.nn.bias_add(tf.matmul(mentee_conv7_flat, weights), biases)
                with tf.name_scope('mentor_embed_8'):
                        shape = int(np.prod(data_dict_teacher.conv3_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentor_conv8_flat = tf.reshape(data_dict_teacher.conv3_1, [-1, shape])
                        embed_mentor_8 = tf.nn.bias_add(tf.matmul(mentor_conv8_flat, weights), biases)
                
                with tf.name_scope('mentee_embed_8'):
                        shape = int(np.prod(data_dict_student.conv8_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentee_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentee_conv8_flat = tf.reshape(data_dict_student.conv8_1, [-1, shape])
                        embed_mentee_8 = tf.nn.bias_add(tf.matmul(mentee_conv8_flat, weights), biases)
                        
                with tf.name_scope('mentor_embed_9'):
                        shape = int(np.prod(data_dict_teacher.conv4_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentor_conv9_flat = tf.reshape(data_dict_teacher.conv4_1, [-1, shape])
                        embed_mentor_9 = tf.nn.bias_add(tf.matmul(mentor_conv9_flat, weights), biases)
                
                with tf.name_scope('mentee_embed_9'):
                        shape = int(np.prod(data_dict_student.conv11_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentee_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentee_conv9_flat = tf.reshape(data_dict_student.conv11_1, [-1, shape])
                        embed_mentee_9 = tf.nn.bias_add(tf.matmul(mentee_conv9_flat, weights), biases)
                with tf.name_scope('mentor_embed_10'):
                        shape = int(np.prod(data_dict_teacher.conv5_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentor_conv10_flat = tf.reshape(data_dict_teacher.conv5_1, [-1, shape])
                        embed_mentor_10 = tf.nn.bias_add(tf.matmul(mentor_conv10_flat, weights), biases)
                
                with tf.name_scope('mentee_embed_10'):
                        shape = int(np.prod(data_dict_student.conv14_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentee_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentee_conv10_flat = tf.reshape(data_dict_student.conv14_1, [-1, shape])
                        embed_mentee_10 = tf.nn.bias_add(tf.matmul(mentee_conv10_flat, weights), biases)
                        
                with tf.name_scope('mentor_embed_11'):
                        shape = int(np.prod(data_dict_teacher.conv6_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentor_conv11_flat = tf.reshape(data_dict_teacher.conv6_1, [-1, shape])
                        embed_mentor_11 = tf.nn.bias_add(tf.matmul(mentor_conv11_flat, weights), biases)
                
                with tf.name_scope('mentee_embed_11'):
                        shape = int(np.prod(data_dict_student.conv17_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentee_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentee_conv11_flat = tf.reshape(data_dict_student.conv17_1, [-1, shape])
                        embed_mentee_11 = tf.nn.bias_add(tf.matmul(mentee_conv11_flat, weights), biases)
                        
                       
                self.loss_embed_3 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_3, embed_mentee_3))))
                self.loss_embed_4 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_4, embed_mentee_4))))
                self.loss_embed_5 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_5, embed_mentee_5))))
                
                self.loss_embed_6 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_6, embed_mentee_6))))
                self.loss_embed_7 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_7, embed_mentee_7))))
                self.loss_embed_8 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_8, embed_mentee_8))))
                self.loss_embed_9 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_9, embed_mentee_9))))
                self.loss_embed_10 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_10, embed_mentee_10))))
                self.loss_embed_11 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_11, embed_mentee_11))))

                return self
