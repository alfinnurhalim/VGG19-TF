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

                        mentor_conv5_flat =tf.reshape(data_dict_teacher.conv1_1, [-1, shape])
                        embed_mentor_2 = tf.nn.bias_add(tf.matmul(mentor_conv5_flat, weights), biases)

                with tf.name_scope('mentee_embed_3'):
                        shape = int(np.prod(data_dict_student.conv1_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentee_embed')

                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentee_conv2_flat = tf.reshape(data_dict_student.conv1_1, [-1, shape])
                        embed_mentee_2 = tf.nn.bias_add(tf.matmul(mentee_conv2_flat, weights), biases)
                        

                with tf.name_scope('mentor_embed_2'):
                        shape = int(np.prod(data_dict_teacher.conv2_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), name = 'biases_mentor_embed', trainable = True)

                        mentor_conv5_flat =tf.reshape(data_dict_teacher.conv2_1, [-1, shape])
                        embed_mentor_2 = tf.nn.bias_add(tf.matmul(mentor_conv5_flat, weights), biases)

                with tf.name_scope('mentee_embed_2'):
                        shape = int(np.prod(data_dict_student.conv2_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentee_embed')

                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentee_conv2_flat = tf.reshape(data_dict_student.conv2_1, [-1, shape])
                        embed_mentee_2 = tf.nn.bias_add(tf.matmul(mentee_conv2_flat, weights), biases)
                        

                with tf.name_scope('mentor_embed_3'):
                        shape = int(np.prod(data_dict_teacher.conv3_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), name = 'biases_mentor_embed', trainable = True)

                        mentor_conv5_flat =tf.reshape(data_dict_teacher.conv3_1, [-1, shape])
                        embed_mentor_2 = tf.nn.bias_add(tf.matmul(mentor_conv5_flat, weights), biases)

                with tf.name_scope('mentee_embed_3'):
                        shape = int(np.prod(data_dict_student.conv3_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentee_embed')

                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentee_conv2_flat = tf.reshape(data_dict_student.conv3_1, [-1, shape])
                        embed_mentee_2 = tf.nn.bias_add(tf.matmul(mentee_conv2_flat, weights), biases)
                        
                self.loss_embed_2 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_2, embed_mentee_2))))
                self.loss_embed_3 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_2, embed_mentee_2))))
                self.loss_embed_4 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_2, embed_mentee_2))))

                return self
