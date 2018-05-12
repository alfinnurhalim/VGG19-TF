import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import math
import numpy as np
import pdb

## No of units utilized to connect mentor-mentee layers of different widths.
EMBED_UNITS = 64


## Embed class consists of embed layers which connect mentor and mentee layers of different widths.
class Embed(object):

	def __init__(self, trainable=True, dropout=0.5):
		self.trainable = trainable
		self.dropout = dropout

	def build(self, data_dict_teacher, data_dict_student, train_mode):
            if(train_mode == 'HT'):
                with tf.name_scope('mentor_embed_1'):
                        ## data_dict_teacher: It is a dictionary containing outputs of all the layers of the teacher class
                        ## data_dict_student: It is a dictionary containing outputs of all the layers of the student class
                        ## trainable: it is a boolean value set to true if the layer needs to be trainable else set to false.
                        shape = int(np.prod(data_dict_teacher.pool2.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True, name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentor_embed')
                        ## flatten the output of teacher's pool2 layer
                        mentor_conv1_flat = tf.reshape(data_dict_teacher.pool2, [-1, shape])

                        embed_mentor_1 = tf.nn.bias_add(tf.matmul(mentor_conv1_flat, weights), biases)

                
                with tf.name_scope('mentee_embed_1'):

                        
                        shape = int(np.prod(data_dict_student.pool2.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentee_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True, name = 'biases_mentee_embed')
                        ## flatten the output of student's pool2 layer

                        ## the reason for flattening teacher's and student's pool2 layer is to have same shapes so that loss_embed_1 can be calculated 
                        ## Here the embed unit is a fully connected layer with the student pool2 layer
                        mentee_conv1_flat = tf.reshape(data_dict_student.pool2, [-1, shape])
                        embed_mentee_1 = tf.nn.bias_add(tf.matmul(mentee_conv1_flat, weights), biases)
                
                ## loss_embed_1: Root mean square error loss between embed_mentor_1 and embed_mentee_1;
                ## embed_mentor_1 is the output of mentor's/teacher's  pool2 layer 
                ## embed_mentee_1 is the output of mentee's/student's  pool2 layer
                loss_embed_1 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_1, embed_mentee_1))))
                
                return loss_embed_1

            elif(train_mode == 'ML'):
               
                ## Mapping of teacher-student layers as follows:: 1st layer of teacher -> 1st layer of student; 2nd layer of teacher -> 2nd layer of student; 3rd layer of teacher -> 3rd of student
                ## output of teacher's conv1_1 layer
                with tf.name_scope('mentor_embed_3'):
                        shape = int(np.prod(data_dict_teacher.conv1_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), name = 'biases_mentor_embed', trainable = True)

                        mentor_conv3_flat =tf.reshape(data_dict_teacher.conv1_1, [-1, shape])
                        
                        embed_mentor_3 = tf.nn.bias_add(tf.matmul(mentor_conv3_flat, weights), biases)
                ## output of student's conv1_1 layer
                with tf.name_scope('mentee_embed_3'):
                        shape = int(np.prod(data_dict_student.conv1_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentee_embed')

                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentee_conv3_flat = tf.reshape(data_dict_student.conv1_1, [-1, shape])
                        embed_mentee_3 = tf.nn.bias_add(tf.matmul(mentee_conv3_flat, weights), biases)
                        
                ## output of teacher's conv2_1 layer
                with tf.name_scope('mentor_embed_4'):
                        shape = int(np.prod(data_dict_teacher.conv2_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), name = 'biases_mentor_embed', trainable = True)

                        mentor_conv4_flat =tf.reshape(data_dict_teacher.conv2_1, [-1, shape])
                        embed_mentor_4 = tf.nn.bias_add(tf.matmul(mentor_conv4_flat, weights), biases)
                ## output of student's conv2_1 layer
                with tf.name_scope('mentee_embed_4'):
                        shape = int(np.prod(data_dict_student.conv2_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentee_embed')

                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentee_conv4_flat = tf.reshape(data_dict_student.conv2_1, [-1, shape])
                        embed_mentee_4 = tf.nn.bias_add(tf.matmul(mentee_conv4_flat, weights), biases)
                        
                
                ## output of teacher's conv3_1 layer
                with tf.name_scope('mentor_embed_5'):
                        shape = int(np.prod(data_dict_teacher.conv3_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), name = 'biases_mentor_embed', trainable = True)
                        mentor_conv5_flat =tf.reshape(data_dict_teacher.conv3_1, [-1, shape])
                        embed_mentor_5 = tf.nn.bias_add(tf.matmul(mentor_conv5_flat, weights), biases)

                ## output of student's conv3_1 layer
                with tf.name_scope('mentee_embed_5'):
                        shape = int(np.prod(data_dict_student.conv3_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentee_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentee_conv5_flat = tf.reshape(data_dict_student.conv3_1, [-1, shape])
                        embed_mentee_5 = tf.nn.bias_add(tf.matmul(mentee_conv5_flat, weights), biases)
                ## teacher-student mapping layers are as follows:: 1st layer of teacher -> 3rd layer of student; 2nd layer of teacher -> 5th layer of student; 3rd layer of teacher -> 8th layer of student; 4th layer of teacher -> 11th layer of student; 5th layer of teacher-> 14th layer of student; 6th layer of teacher -> 17th layer of student

                ## output of teacher's conv1_1 layer
                with tf.name_scope('mentor_embed_6'):
                        shape = int(np.prod(data_dict_teacher.conv1_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentor_conv6_flat = tf.reshape(data_dict_teacher.conv1_1, [-1, shape])
                        embed_mentor_6 = tf.nn.bias_add(tf.matmul(mentor_conv6_flat, weights), biases)
                
                ## output of student's conv3_1 layer
                with tf.name_scope('mentee_embed_6'):
                        shape = int(np.prod(data_dict_student.conv3_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentee_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentee_conv6_flat = tf.reshape(data_dict_student.conv3_1, [-1, shape])
                        embed_mentee_6 = tf.nn.bias_add(tf.matmul(mentee_conv6_flat, weights), biases)
                ## output of teacher's conv2_1 layer
                with tf.name_scope('mentor_embed_7'):
                        shape = int(np.prod(data_dict_teacher.conv2_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentor_conv7_flat = tf.reshape(data_dict_teacher.conv2_1, [-1, shape])
                        embed_mentor_7 = tf.nn.bias_add(tf.matmul(mentor_conv7_flat, weights), biases)
                
                ## output of student's conv5_1 layer
                with tf.name_scope('mentee_embed_7'):
                        shape = int(np.prod(data_dict_student.conv5_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentee_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentee_conv7_flat = tf.reshape(data_dict_student.conv5_1, [-1, shape])
                        embed_mentee_7 = tf.nn.bias_add(tf.matmul(mentee_conv7_flat, weights), biases)
                ## output of teacher's conv3_1 layer
                with tf.name_scope('mentor_embed_8'):
                        shape = int(np.prod(data_dict_teacher.conv3_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentor_conv8_flat = tf.reshape(data_dict_teacher.conv3_1, [-1, shape])
                        embed_mentor_8 = tf.nn.bias_add(tf.matmul(mentor_conv8_flat, weights), biases)
                
                ## output of student's conv8_1 layer
                with tf.name_scope('mentee_embed_8'):
                        shape = int(np.prod(data_dict_student.conv8_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentee_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentee_conv8_flat = tf.reshape(data_dict_student.conv8_1, [-1, shape])
                        embed_mentee_8 = tf.nn.bias_add(tf.matmul(mentee_conv8_flat, weights), biases)
                        
                ## output of teacher's conv4_1 layer
                with tf.name_scope('mentor_embed_9'):
                        shape = int(np.prod(data_dict_teacher.conv4_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentor_conv9_flat = tf.reshape(data_dict_teacher.conv4_1, [-1, shape])
                        embed_mentor_9 = tf.nn.bias_add(tf.matmul(mentor_conv9_flat, weights), biases)
                
                ## output of student's conv11_1 layer
                with tf.name_scope('mentee_embed_9'):
                        shape = int(np.prod(data_dict_student.conv11_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentee_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentee_conv9_flat = tf.reshape(data_dict_student.conv11_1, [-1, shape])
                        embed_mentee_9 = tf.nn.bias_add(tf.matmul(mentee_conv9_flat, weights), biases)
                ## output of teacher's conv5_1 layer
                with tf.name_scope('mentor_embed_10'):
                        shape = int(np.prod(data_dict_teacher.conv5_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentor_conv10_flat = tf.reshape(data_dict_teacher.conv5_1, [-1, shape])
                        embed_mentor_10 = tf.nn.bias_add(tf.matmul(mentor_conv10_flat, weights), biases)
                
                ## output of student's conv14_1 layer
                with tf.name_scope('mentee_embed_10'):
                        shape = int(np.prod(data_dict_student.conv14_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentee_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentee_conv10_flat = tf.reshape(data_dict_student.conv14_1, [-1, shape])
                        embed_mentee_10 = tf.nn.bias_add(tf.matmul(mentee_conv10_flat, weights), biases)
                        
                ## output of teacher's conv6_1 layer
                with tf.name_scope('mentor_embed_11'):
                        shape = int(np.prod(data_dict_teacher.conv6_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentor_conv11_flat = tf.reshape(data_dict_teacher.conv6_1, [-1, shape])
                        embed_mentor_11 = tf.nn.bias_add(tf.matmul(mentor_conv11_flat, weights), biases)
                
                ## output of student's conv17_1 layer
                with tf.name_scope('mentee_embed_11'):
                        shape = int(np.prod(data_dict_student.conv17_1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentee_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentee_conv11_flat = tf.reshape(data_dict_student.conv17_1, [-1, shape])
                        embed_mentee_11 = tf.nn.bias_add(tf.matmul(mentee_conv11_flat, weights), biases)
                        
                ## Root mean square loss between mentor and mentee embed layers       
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
