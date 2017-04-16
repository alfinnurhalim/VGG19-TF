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

	def build(self, rgb, mentor_conv3, mentor_conv5, mentee_conv1, mentee_conv2, train_mode=None):
                with tf.name_scope('mentor_embed_1'):
                        shape = int(np.prod(mentor_conv3.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True, name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentor_embed')
                        mentor_conv3_flat = tf.reshape(mentor_conv3, [-1, shape])

                        embed_mentor_1 = tf.nn.bias_add(tf.matmul(mentor_conv3_flat, weights), biases)

                
                with tf.name_scope('mentee_embed_1'):
                        shape = int(np.prod(mentee_conv1.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentee_embed')

                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True, name = 'biases_mentee_embed')

                        mentee_conv1_flat = tf.reshape(mentee_conv1, [-1, shape])

                        embed_mentee_1 = tf.nn.bias_add(tf.matmul(mentee_conv1_flat, weights), biases)

                with tf.name_scope('mentor_embed_2'):
                        shape = int(np.prod(mentor_conv5.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), name = 'biases_mentor_embed', trainable = True)

                        mentor_conv5_flat =tf.reshape(mentor_conv5, [-1, shape])
                        embed_mentor_2 = tf.nn.bias_add(tf.matmul(mentor_conv5_flat, weights), biases)

                with tf.name_scope('mentee_embed_2'):
                        shape = int(np.prod(mentee_conv2.get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = True,name = 'weights_mentee_embed')

                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = True,name = 'biases_mentee_embed')
                        mentee_conv2_flat = tf.reshape(mentee_conv2, [-1, shape])
                        embed_mentee_2 = tf.nn.bias_add(tf.matmul(mentee_conv2_flat, weights), biases)
                        
                loss_embed_1 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_1, embed_mentee_1))))
                loss_embed_2 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_2, embed_mentee_2))))

                return loss_embed_1, loss_embed_2
