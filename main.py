import tensorflow as tf 
import random
from DataInput import DataInput
from teacher import Teacher 
from student import Student
import os
import sys
import time
import pdb
import numpy as np
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.python import pywrap_tensorflow
import argparse
from  embed import Embed
dataset_path = "./"
epoch = 0
NUM_CHANNELS = 3
NUM_EPOCHS_PER_DECAY = 1.0
NUM_ITERATIONS = 5000000
FINAL_LEARNING_RATE = 0.0000002
LEARNING_RATE_DECAY_FACTOR = 0.9809
lamda_value = 4.0
SUMMARY_LOG_DIR="./summary-log"
seed = 1234
class FitNet(object):

    def device_and_target(self):
        cluster_spec = tf.train.ClusterSpec({
        "ps": FLAGS.ps_hosts.split(","),
        "worker": FLAGS.worker_hosts.split(","),
        
        })
        server = tf.train.Server(
            cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
        if FLAGS.job_name == "ps":
            server.join()
        
        worker_device = "/job:worker/task:{}".format(FLAGS.task_index)
        return (tf.train.replica_device_setter
                    (worker_device=worker_device,
                        cluster=cluster_spec),
                        server.target,
                        )

         
    def placeholder_inputs(self, batch_size):
            images_placeholder = tf.placeholder(tf.float32, 
                                                                    shape=(FLAGS.batch_size, FLAGS.image_height, 
                                                                               FLAGS.image_width, NUM_CHANNELS))
            labels_placeholder = tf.placeholder(tf.int32,
                                                                    shape=(FLAGS.batch_size))

            return images_placeholder, labels_placeholder

    def fill_feed_dict(self, data_input, images_pl, labels_pl,sess, phase_train, mode):
            images_feed, labels_feed = sess.run([data_input.example_batch, data_input.label_batch])
            if mode == 'Train':
                feed_dict = {
                        images_pl: images_feed,
                        labels_pl: labels_feed,
                        phase_train:True
                }
            if mode == 'Test':
                feed_dict = {
                        images_pl: images_feed,
                        labels_pl: labels_feed,
                        phase_train:False
                }
            return feed_dict

    def do_eval(self, sess,
                            eval_correct,
                            logits,
                            images_placeholder,
                            labels_placeholder,
                            dataset, phase_train, mode):

            true_count = 0
            steps_per_epoch = dataset.num_examples //FLAGS.batch_size 
            num_examples = steps_per_epoch * FLAGS.batch_size
            for step in xrange(steps_per_epoch):
                    feed_dict = self.fill_feed_dict(dataset, images_placeholder,
                                                                            labels_placeholder,sess, phase_train, mode)

                    count = sess.run(eval_correct, feed_dict=feed_dict)
                    true_count = true_count + count
            precision = float(true_count) / num_examples
            print ('  Num examples: %d, Num correct: %d, Precision @ 1: %0.04f' %
                            (num_examples, true_count, precision))

    def evaluation(self, logits, labels):
            correct = tf.nn.in_top_k(logits, labels, 1)
            return tf.reduce_sum(tf.cast(correct, tf.int32))

    def train_op_for_multiple_optimizers(self, lr, loss, data_dict_embed):

            l1_var_list = []
            l2_var_list = []
            l3_var_list = []
            l4_var_list = []
            l5_var_list = []
            l6_var_list = []

            if FLAGS.multiple_optimizers_l1:
                self.train_op0 = tf.train.AdamOptimizer(lr).minimize(loss)
                self.train_op1 = tf.train.AdamOptimizer(lr).minimize(data_dict_embed.loss_embed_3, var_list = self.get_layer_1_variables_of_student(l1_var_list))

            elif FLAGS.multiple_optimizers_l2:
                self.train_op0 = tf.train.AdamOptimizer(lr).minimize(loss)
                self.train_op1 = tf.train.AdamOptimizer(lr).minimize(data_dict_embed.loss_embed_3, var_list= self.get_layer_1_variables_of_student(l1_var_list))
                self.train_op2 = tf.train.AdamOptimizer(lr).minimize(data_dict_embed.loss_embed_4, var_list= self.get_layer_2_variables_of_student(l2_var_list))
            
            elif FLAGS.multiple_optimizers_l3:
                self.train_op0 = tf.train.AdamOptimizer(lr).minimize(loss)
                self.train_op1 = tf.train.AdamOptimizer(lr).minimize(data_dict_embed.loss_embed_3, var_list=self.get_layer_1_variables_of_student(l1_var_list))
                self.train_op2 = tf.train.AdamOptimizer(lr).minimize(data_dict_embed.loss_embed_4, var_list=self.get_layer_2_variables_of_student(l2_var_list))
                self.train_op3 = tf.train.AdamOptimizer(lr).minimize(data_dict_embed.loss_embed_5, var_list=self.get_layer_3_variables_of_student(l3_var_list))
        
            elif FLAGS.multiple_optimizers_metric:
                self.train_op0 = tf.train.AdamOptimizer(lr).minimize(loss)
                self.train_op1 = tf.train.AdamOptimizer(lr).minimize(data_dict_embed.loss_embed_6, var_list=self.get_layer_3_variables_of_student(l1_var_list))
                self.train_op2 = tf.train.AdamOptimizer(lr).minimize(data_dict_embed.loss_embed_7, var_list=self.get_layer_5_variables_of_student(l2_var_list))
                self.train_op3 = tf.train.AdamOptimizer(lr).minimize(data_dict_embed.loss_embed_8, var_list=self.get_layer_8_variables_of_student(l3_var_list))
                self.train_op4 = tf.train.AdamOptimizer(lr).minimize(data_dict_embed.loss_embed_9, var_list=self.get_layer_11_variables_of_student(l4_var_list))
                self.train_op5 = tf.train.AdamOptimizer(lr).minimize(data_dict_embed.loss_embed_10, var_list=self.get_layer_14_variables_of_student(l5_var_list))
                self.train_op6 = tf.train.AdamOptimizer(lr).minimize(data_dict_embed.loss_embed_11, var_list=self.get_layer_17_variables_of_student(l6_var_list))
            elif FLAGS.multiple_optimizers_1_17:
                self.train_op0 = tf.train.AdamOptimizer(lr).minimize(loss)
                self.train_op1 = tf.train.AdamOptimizer(lr).minimize(data_dict_embed.loss_embed_3, var_list=self.get_layer_1_variables_of_student(l1_var_list))
                self.train_op2 = tf.train.AdamOptimizer(lr).minimize(data_dict_embed.loss_embed_11, var_list=self.get_layer_17_variables_of_student(l2_var_list))

            elif FLAGS.multiple_optimizers_random:
                self.train_op0 = tf.train.AdamOptimizer(lr).minimize(loss)
                self.train_op1 = tf.train.AdamOptimizer(lr).minimize(data_dict_embed.loss_embed_6, var_list=self.get_layer_3_variables_of_student(l1_var_list))
                self.train_op2 = tf.train.AdamOptimizer(lr).minimize(data_dict_embed.loss_embed_7, var_list=self.get_layer_5_variables_of_student(l2_var_list))
                self.train_op2 = tf.train.AdamOptimizer(lr).minimize(data_dict_embed.loss_embed_8, var_list=self.get_layer_8_variables_of_student(l3_var_list))

    def calculate_loss_with_multiple_optimizers(self, loss, data_dict_embed, sess, feed_dict):

            if FLAGS.multiple_optimizers_l1:
                _, self.loss_value0 = sess.run([self.train_op0, loss], feed_dict=feed_dict)
                _, self.loss_value1 = sess.run([self.train_op1, data_dict_embed.loss_embed_3], feed_dict=feed_dict)
            elif FLAGS.multiple_optimizers_l2:
                _, self.loss_value0 = sess.run([self.train_op0, loss], feed_dict=feed_dict)
                _, self.loss_value1 = sess.run([self.train_op1, data_dict_embed.loss_embed_3], feed_dict=feed_dict)
                _, self.loss_value2 = sess.run([self.train_op2, data_dict_embed.loss_embed_4], feed_dict=feed_dict)
            elif FLAGS.multiple_optimizers_l3:
                _, self.loss_value0 = sess.run([self.train_op0, loss], feed_dict=feed_dict)
                _, self.loss_value1 = sess.run([self.train_op1, data_dict_embed.loss_embed_3], feed_dict=feed_dict)
                _, self.loss_value2 = sess.run([self.train_op2, data_dict_embed.loss_embed_4], feed_dict=feed_dict)
                _, self.loss_value3 = sess.run([self.train_op3, data_dict_embed.loss_embed_5], feed_dict=feed_dict)

            elif FLAGS.multiple_optimizers_metric:
                _, self.loss_value0 = sess.run([self.train_op0, loss], feed_dict=feed_dict)
                _, self.loss_value1 = sess.run([self.train_op1, data_dict_embed.loss_embed_6], feed_dict=feed_dict)
                _, self.loss_value2 = sess.run([self.train_op2, data_dict_embed.loss_embed_7], feed_dict=feed_dict)
                _, self.loss_value3 = sess.run([self.train_op3, data_dict_embed.loss_embed_8], feed_dict=feed_dict)
                _, self.loss_value4 = sess.run([self.train_op4, data_dict_embed.loss_embed_9], feed_dict=feed_dict)
                _, self.loss_value5 = sess.run([self.train_op5, data_dict_embed.loss_embed_10], feed_dict=feed_dict)
                _, self.loss_value6 = sess.run([self.train_op6, data_dict_embed.loss_embed_11], feed_dict=feed_dict)

            elif FLAGS.multiple_optimizers_1_17:
                _, self.loss_value0 = sess.run([self.train_op0, loss], feed_dict=feed_dict)
                _, self.loss_value1 = sess.run([self.train_op1, data_dict_embed.loss_embed_3], feed_dict=feed_dict)
                _, self.loss_value2 = sess.run([self.train_op2, data_dict_embed.loss_embed_11], feed_dict=feed_dict)

            elif FLAGS.multiple_optimizers_random:
                _, self.loss_value0 = sess.run([self.train_op0, loss], feed_dict=feed_dict)
                _, self.loss_value1 = sess.run([self.train_op1, data_dict_embed.loss_embed_6], feed_dict=feed_dict)
                _, self.loss_value2 = sess.run([self.train_op2, data_dict_embed.loss_embed_7], feed_dict=feed_dict)
                _, self.loss_value2 = sess.run([self.train_op2, data_dict_embed.loss_embed_8], feed_dict=feed_dict)

    def get_variables_to_restore(self, variables_to_restore):

            variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="teacher_conv1_1/weights"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "teacher_conv1_1/biases:0"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "teacher_conv2_1/weights:0"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "teacher_conv2_1/biases:0"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "teacher_conv3_1/weights:0"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "teacher_conv3_1/biases:0"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "teacher_fc1/weights:0"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "teacher_fc1/biases:0"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "teacher_fc2/weights:0"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "teacher_fc2/biases:0"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "teacher_fc3/weights:0"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "teacher_fc3/biases:0"][0])

            return variables_to_restore

    def get_variables_to_restore_KD(self, variables_to_restore):

            variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="student_conv1_1/weights"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv1_1/biases:0"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv2_1/weights:0"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv2_1/biases:0"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv3_1/weights:0"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv3_1/biases:0"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv4_1/weights:0"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv4_1/biases:0"][0])
            variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="student_conv5_1/weights"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv5_1/biases:0"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv6_1/weights:0"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv6_1/biases:0"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv7_1/weights:0"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv7_1/biases:0"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv8_1/weights:0"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv8_1/biases:0"][0])
            variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="student_conv9_1/weights"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv9_1/biases:0"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv10_1/weights:0"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv10_1/biases:0"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv11_1/weights:0"][0])
            variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv11_1/biases:0"][0])

            return variables_to_restore

    def get_layer_1_variables_of_student(self, l1_variables):

            l1_variables.append([var for var in tf.global_variables() if var.op.name=="student_conv1_1/weights"][0])
            l1_variables.append([v for v in tf.global_variables() if v.name == "student_conv1_1/biases:0"][0])
            return l1_variables


    def get_layer_2_variables_of_student(self, l2_variables):

            l2_variables.append([var for var in tf.global_variables() if var.op.name=="student_conv2_1/weights"][0])
            l2_variables.append([v for v in tf.global_variables() if v.name == "student_conv2_1/biases:0"][0])

            return l2_variables

    def get_layer_3_variables_of_student(self, l3_variables):

            l3_variables.append([var for var in tf.global_variables() if var.op.name=="student_conv3_1/weights"][0])
            l3_variables.append([v for v in tf.global_variables() if v.name == "student_conv3_1/biases:0"][0])
            return l3_variables

    def get_layer_5_variables_of_student(self, l5_variables):

            l5_variables.append([var for var in tf.global_variables() if var.op.name=="student_conv5_1/weights"][0])
            l5_variables.append([v for v in tf.global_variables() if v.name == "student_conv5_1/biases:0"][0])
            return l5_variables
    def get_layer_8_variables_of_student(self, l8_variables):

            l8_variables.append([var for var in tf.global_variables() if var.op.name=="student_conv8_1/weights"][0])
            l8_variables.append([v for v in tf.global_variables() if v.name == "student_conv8_1/biases:0"][0])
            return l8_variables
    def get_layer_11_variables_of_student(self, l11_variables):

            l11_variables.append([var for var in tf.global_variables() if var.op.name=="student_conv11_1/weights"][0])
            l11_variables.append([v for v in tf.global_variables() if v.name == "student_conv11_1/biases:0"][0])
            return l11_variables
    def get_layer_14_variables_of_student(self, l14_variables):

            l14_variables.append([var for var in tf.global_variables() if var.op.name=="student_conv14_1/weights"][0])
            l14_variables.append([v for v in tf.global_variables() if v.name == "student_conv14_1/biases:0"][0])
            return l14_variables
    def get_layer_17_variables_of_student(self, l17_variables):

            l17_variables.append([var for var in tf.global_variables() if var.op.name=="student_conv17_1/weights"][0])
            l17_variables.append([v for v in tf.global_variables() if v.name == "student_conv17_1/biases:0"][0])
            return l17_variables

    def train_independent_teacher(self, images_placeholder, labels_placeholder, global_step, keep_prob, sess):
            print("Teacher")
            teacher = Teacher(True)
            num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
            decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
            data_dict_teacher = teacher.build(images_placeholder, keep_prob, FLAGS.num_classes)
            self.loss = teacher.loss(labels_placeholder)

            lr = tf.train.exponential_decay(FLAGS.learning_rate,global_step, decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)
            self.train_op = teacher.training(self.loss, lr, global_step)
            self.softmax = data_dict_teacher.softmax_output
            init = tf.initialize_all_variables()
            sess.run(init)
            saver = tf.train.Saver()

    def train_student_with_hint_based_training(self, images_placeholder, labels_placeholder, global_step, keep_prob, sess):
            print("Student with Hind based approach")
            teacher = Teacher(False)
            student = Student(True)
            data_dict_teacher = teacher.build(images_placeholder, keep_prob, FLAGS.num_classes)
            data_dict_student = student.build(images_placeholder, FLAGS.num_classes)
            embed = Embed()
            loss = embed.build(data_dict_teacher.pool2, data_dict_student.pool2, 'HT')
            variables_to_restore = []
            variables_to_restore = self.get_variables_to_restore(variables_to_restore)
            saver = tf.train.Saver(variables_to_restore)
            num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
            decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
            lr = tf.train.exponential_decay(FLAGS.learning_rate,global_step, decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)
            optimizer= tf.train.AdamOptimizer(lr)
            variables_to_restore = []
            train_op = optimizer.minimize(loss, var_list= get_variables_to_restore_KD(variables_to_restore))
            self.softmax = data_dict_student.softmax_output
            init = tf.initialize_all_variables()
            sess.run(init)
            saver.restore(sess, FLAGS.teacher_weights_filename)

    def train_student_with_knowledge_distillation(self, images_placeholder, labels_placeholder, global_step, keep_prob, sess):

            print("Student with Knowledge Distillation Approach")
            teacher = Teacher(False)
            student = Student(True)
            data_dict_teacher = teacher.build(images_placeholder, keep_prob, FLAGS.num_classes)
            data_dict_student = student.build(images_placeholder, FLAGS.num_classes)
            embed = Embed()
            variables_to_restore = []
            variables_to_restore = self.get_variables_to_restore(variables_to_restore)
            #softmax_loss = embed.build(teacher_softmax_layer_loss, student_softmax_layer_loss, 'KD')
            saver = tf.train.Saver(variables_to_restore)
            softmax_loss = tf.reduce_mean(tf.square(tf.subtract(data_dict_student.softmax_output, data_dict_teacher.softmax_output))) 
            loss = student.loss(labels_placeholder)
            global_step = tf.Variable(0, trainable=False)
            lamda = tf.train.exponential_decay(lamda_value, global_step, 10000, 1.0, staircase = True)
            num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
            decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
            lr = tf.train.exponential_decay(FLAGS.learning_rate,global_step, decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)
            total_loss = loss + lamda*softmax_loss 
            self.train_op = student.training(total_loss, lr)
            self.softmax = data_dict_student.softmax_output
            init = tf.initialize_all_variables()
            sess.run(init)
            variables_to_restore = []
            saver_KD = tf.train.Saver(get_variables_to_restore_KD(variables_to_restore))
            saver_KD.restore(sess, FLAGS.HT_filename)
            saver.restore(sess, "./summary-log/teacher_weights_filename_cifar10")

    def train_student_with_hard_logits(self, images_placeholder, labels_placeholder, global_step, keep_prob, sess):
            print("Student with hard logits approach")
            teacher = Teacher(False)
            student = Student(True)
            data_dict_teacher = teacher.build(images_placeholder, keep_prob, FLAGS.num_classes)
            ind_max = tf.argmax(data_dict_teacher.logits_temp, axis = 1)
            hard_logits = tf.one_hot(ind_max, FLAGS.num_classes)

            data_dict_student = student.build(images_placeholder, FLAGS.num_classes)
            embed = Embed()
            variables_to_restore = []
            variables_to_restore = self.get_variables_to_restore(variables_to_restore)
            #softmax_loss = embed.build(teacher_softmax_layer_loss, student_softmax_layer_loss, 'KD')
            saver = tf.train.Saver(variables_to_restore)
            softmax_loss = tf.reduce_mean(tf.square(tf.subtract(data_dict_student.softmax_output, hard_logits))) 
            self.loss = student.loss(labels_placeholder)
            global_step = tf.Variable(0, trainable=False)
            lamda = tf.train.exponential_decay(lamda_value, global_step, 10000, 1.0, staircase = True)
            num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
            decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
            lr = tf.train.exponential_decay(FLAGS.learning_rate,global_step, decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)
            total_loss = softmax_loss 
            train_op = student.training(total_loss, lr)
            self.softmax = data_dict_student.softmax_output
            init = tf.initialize_all_variables()
            sess.run(init)
            saver.restore(sess, "./summary-log/teacher_weights_filename_caltech101")

    def train_student_with_multiple_layers(self, images_placeholder, labels_placeholder, global_step, keep_prob, sess):

            print("Student with multiple layers approach")
            teacher = Teacher(False)
            student = Student(True)
            data_dict_teacher = teacher.build(images_placeholder, keep_prob, FLAGS.num_classes)
            data_dict_student = student.build(images_placeholder, FLAGS.num_classes)
            embed = Embed()
            data_dict_embed = embed.build(data_dict_teacher, data_dict_student, 'ML')
            variables_to_restore = []
            variables_to_restore = self.get_variables_to_restore(variables_to_restore)
            saver = tf.train.Saver(variables_to_restore)
            num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
            decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
            lr = tf.train.exponential_decay(FLAGS.learning_rate,global_step, decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)
            self.loss = student.loss(labels_placeholder)
            optimizer= tf.train.AdamOptimizer(lr)
            variables_to_restore = []
            #train_op = optimizer.minimize(loss + data_dict_embed.loss_embed_2 + data_dict_embed.loss_embed_3 + data_dict_embed.loss_embed_4)
            #train_op = optimizer.minimize(loss + data_dict_embed.loss_embed_3 + data_dict_embed.loss_embed_4 + data_dict_embed.loss_embed_5)
            self.train_op = optimizer.minimize(self.loss + data_dict_embed.loss_embed_6 + data_dict_embed.loss_embed_7 + data_dict_embed.loss_embed_8)
            self.softmax = data_dict_student.softmax_output
            init = tf.initialize_all_variables()
            sess.run(init)
            saver.restore(sess, FLAGS.teacher_weights_filename)


    def train_student_with_new_method(self, images_placeholder, labels_placeholder, global_step, keep_prob, sess):

            print("Student with new method approach")
            teacher = Teacher(False)
            student = Student(True)
            data_dict_teacher = teacher.build(images_placeholder, keep_prob, FLAGS.num_classes)
            data_dict_student = student.build(images_placeholder, FLAGS.num_classes)
            embed = Embed()
            data_dict_embed = embed.build(data_dict_teacher, data_dict_student, 'ML')
            variables_to_restore = []
            variables_to_restore = self.get_variables_to_restore(variables_to_restore)
            saver = tf.train.Saver(variables_to_restore)
            num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
            decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
            lr = tf.train.exponential_decay(FLAGS.learning_rate,global_step, decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)
            self.loss = student.loss(labels_placeholder)
            self.softmax = data_dict_student.softmax_output
            self.train_op_for_multiple_optimizers(lr, self.loss, data_dict_embed)
            init = tf.initialize_all_variables()
            sess.run(init)

    def train_independent_student(self, images_placeholder, labels_placeholder, global_step, keep_prob, sess):
            print("Independent student")
            student = Student(True)
            num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
            decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
            data_dict_student = student.build(images_placeholder, FLAGS.num_classes)
            self.loss = student.loss(labels_placeholder)
            lr = tf.train.exponential_decay(FLAGS.learning_rate,global_step, decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)
            self.train_op = student.training(self.loss, lr)
            self.softmax = data_dict_student.softmax_output
            init = tf.initialize_all_variables()
            sess.run(init)


    def training_and_inference(self, sess, images_placeholder,
            labels_placeholder, data_input_train, data_input_test, phase_train):
            
            try:
                for i in range(NUM_ITERATIONS):
                    feed_dict = self.fill_feed_dict(data_input_train, images_placeholder,
                                                                    labels_placeholder, sess, phase_train, 'Train')
                    if FLAGS.student or FLAGS.teacher:
                        _, loss_value = sess.run([self.train_op, self.loss], feed_dict=feed_dict)                
                        if i % 5 == 0:
                            print ('Step %d: loss = %.5f' % (i, loss_value))
                            #summary_str = sess.run(summary, feed_dict=feed_dict)
                            #summary_writer.add_summary(summary_str, i)
                            #summary_writer.flush()
                                    
                    if FLAGS.dependent_student and FLAGS.multiple_optimizers:
                        self.calculate_loss_with_multiple_optimizers(self.loss, data_dict_embed, sess, feed_dict)
                        if i % 10 == 0:
                            if FLAGS.multiple_optimizers_l1:
                                print('Step %d: loss_value0 = %.20f' % (i, self.loss_value0))
                            elif FLAGS.multiple_optimizers_l2:
                                print('Step %d: loss_value0 = %.20f' % (i, self.loss_value0))
                                print('Step %d: loss_value0 = %.20f' % (i, self.loss_value1))

                            elif FLAGS.multiple_optimizers_l3:

                                print('Step %d: loss_value0 = %.20f' % (i, self.loss_value0))
                                print('Step %d: loss_value0 = %.20f' % (i, self.loss_value1))
                                print('Step %d: loss_value0 = %.20f' % (i, self.loss_value2))

                            elif FLAGS.multiple_optimizers_metric:

                                print('Step %d: loss_value0 = %.20f' % (i, self.loss_value0))
                                print('Step %d: loss_value1 = %.20f' % (i, self.loss_value1))
                                print('Step %d: loss_value2 = %.20f' % (i, self.loss_value2))
                                print('Step %d: loss_value3 = %.20f' % (i, self.loss_value3))
                                print('Step %d: loss_value4 = %.20f' % (i, self.loss_value4))
                                print('Step %d: loss_value5 = %.20f' % (i, self.loss_value5))
                            elif FLAGS.multiple_optimizers_1_17:
                                print('Step %d: loss_value0 = %.20f' % (i, self.loss_value0))
                                print('Step %d: loss_value0 = %.20f' % (i, self.loss_value1))
                                print('Step %d: loss_value0 = %.20f' % (i, self.loss_value2))

                            elif FLAGS.multiple_optimizers_random:
                                print('Step %d: loss_value0 = %.20f' % (i, self.loss_value0))
                                print('Step %d: loss_value0 = %.20f' % (i, self.loss_value1))
                                print('Step %d: loss_value0 = %.20f' % (i, self.loss_value2))
                    if (i) % (FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN//FLAGS.batch_size) == 0 or (i) == NUM_ITERATIONS:
                        global epoch 
                        epoch = epoch + 1
                        checkpoint_file = os.path.join(SUMMARY_LOG_DIR, 'model.ckpt')
                        if FLAGS.teacher:
                            saver = tf.train.Saver()
                            saver.save(sess, FLAGS.teacher_weights_filename)
                                            
                        #elif FLAGS.student and FLAGS.HT:
                            #saver = tf.train.Saver()
                            #saver.save(sess, FLAGS.HT_filename)
                        """
                        elif FLAGS.student and FLAGS.KD:
                            saver.save(sess, FLAGS.KD_filename)

                        elif FLAGS.student:
                            saver.save(sess, FLAGS.student_filename)
                        """
                        print ("Training Data Eval:")
                        self.do_eval(sess,
                            self.eval_correct,
                            self.softmax,
                            images_placeholder,
                            labels_placeholder,
                            data_input_train,
                            phase_train,
                            'Train')
                        print ("Testing Data Eval: EPOCH->", epoch)
                        self.do_eval(sess,
                            self.eval_correct,
                            self.softmax,
                            images_placeholder,
                            labels_placeholder,
                            data_input_test,
                            phase_train,
                            'Test')
                                            
                                
                coord.request_stop()
                coord.join(threads)
            except Exception as e:
                print(e)

    def main(self, _):

            with tf.Graph().as_default():

                    config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth = True))
                    tf.set_random_seed(seed)
                    data_input_train = DataInput(dataset_path, FLAGS.train_dataset, FLAGS.batch_size, FLAGS.image_width, FLAGS.image_height, FLAGS.num_training_examples)
                    data_input_test = DataInput(dataset_path, FLAGS.test_dataset,FLAGS.batch_size, FLAGS.image_width, FLAGS.image_height, FLAGS.num_testing_examples)
                    images_placeholder, labels_placeholder = self.placeholder_inputs(FLAGS.batch_size)

                    summary = tf.summary.merge_all()
                    sess = tf.Session(config=config)
                    summary_writer = tf.summary.FileWriter(SUMMARY_LOG_DIR, sess.graph)
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                    keep_prob = tf.constant(0.5, dtype= tf.float32)
                    phase_train = tf.placeholder(tf.bool, name = 'phase_train')
                    global_step = tf.Variable(0, name='global_step', trainable=False)
                    if FLAGS.teacher:
                        self.train_independent_teacher(images_placeholder, labels_placeholder, global_step, keep_prob, sess)
                        
                    elif (FLAGS.student and FLAGS.HT):
                        self.train_student_with_hint_based_training(images_placeholder, labels_placeholder, global_step, keep_prob, sess)

                    elif (FLAGS.student and FLAGS.KD):
                        self.train_student_with_knowledge_distillation(images_placeholder, labels_placeholder, global_step, keep_prob, sess)

                    elif (FLAGS.student and FLAGS.hard_logits):
                        self.train_student_with_hard_logits(images_placeholder, labels_placeholder, global_step, keep_prob, sess)

                    elif (FLAGS.student and FLAGS.multiple_layers):
                        self.train_student_with_multiple_layers(images_placeholder, labels_placeholder, global_step, keep_prob, sess)

                    elif (FLAGS.dependent_student and FLAGS.new_method):
                        self.train_student_with_new_method(images_placeholder, labels_placeholder, global_step, keep_prob, sess)

                    elif FLAGS.student:
                        self.train_independent_student(images_placeholder, labels_placeholder, global_step, keep_prob, sess)

                    self.eval_correct = self.evaluation(self.softmax, labels_placeholder)

                    self.training_and_inference(sess, images_placeholder,
                                                         labels_placeholder, data_input_train, data_input_test, phase_train)
                                
            sess.close()

if __name__ == '__main__':

        parser = argparse.ArgumentParser()

        parser.add_argument(
        '--HT',
        type=bool,
        help = 'HT is hint based training',
        default = False)

        parser.add_argument(
        '--KD',
        type = bool,
        help = 'Knowledge based training',
        default = False)

        parser.add_argument(
        '--teacher',
        type = bool,
        help = 'train teacher',
        default = False)

        parser.add_argument(
        '--student',
        type = bool,
        help = 'train student with KG or HT',
        default = False
        )
        parser.add_argument(
        '--dependent_student',
        type = bool,
        help = 'dependent_student',
        default = False)

        parser.add_argument(
        '--HT_filename',
        type = str,
        default = "./summary-log/HT_filename_cifar10")

        parser.add_argument(
        '--KD_filename',
        type = str,
        default = "./summary-log/KD_filename")
        
        parser.add_argument(
        '--dataset',
        type = str,
        default = "cifar10")

        parser.add_argument(
        '--test_dataset',
        type = str,
        default = "./Datasets/test_map.txt")

        parser.add_argument(
        '--train_dataset',
        type = str,
        default = "./Datasets/train_map.txt")

        parser.add_argument(
        '--num_training_examples',
        type = int,
        default = 50000)

        parser.add_argument(
        '--num_testing_examples',
        type = int,
        default = 10000)

        parser.add_argument(
        '--NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN',
        type = int,
        default = 50000)

        parser.add_argument(
        '--num_classes',
        type = int,
        default = 10)

        parser.add_argument(
        '--teacher_weights_filename',
        type = str,
        default = "./summary-log/teacher_weights_filename_cifar10_10layer"
        )
        parser.add_argument(
        '--student_filename',
        type = str,
        default = "./summary-log/student_filename_cifar10"
        )

        parser.add_argument(
        '--learning_rate',
        type = float,
        default = 0.0001
        )

        parser.add_argument(
        '--batch_size',
        type = int,
        default = 128
        
        )
        parser.add_argument(
        '--hard_logits',
        type = bool,
        default = False
        )
        parser.add_argument(
        '--multiple_layers',
        type = bool,
        default = False
        )
        parser.add_argument(
        '--multiple_optimizers',
        type = bool,
        default = False
        )
        parser.add_argument(
        '--multiple_optimizers_l1',
        type = bool,
        default = False
        )
        parser.add_argument(
        '--multiple_optimizers_l2',
        type = bool,
        default = False
        )
        parser.add_argument(
        '--multiple_optimizers_l3',
        type = bool,
        default = False
        )
        parser.add_argument(
        '--multiple_optimizers_l4',
        type = bool,
        default = False
        )
        parser.add_argument(
        '--multiple_optimizers_l5',
        type = bool,
        default = False
        )
        parser.add_argument(
        '--multiple_optimizers_l6',
        type = bool,
        default = False
        )
        parser.add_argument(
        '--ps_hosts',
        type = str,
        default = ""
        )
        parser.add_argument(
        '--worker_hosts',
        type = str,
        default = ""
        )
        parser.add_argument(
        '--job_name',
        type = str,
        default = ""
        )
        parser.add_argument(
        '--task_index',
        type = int,
        default =0
        )
        parser.add_argument(
        '--image_width',
        type = int,
        default =32
        )
        parser.add_argument(
        '--image_height',
        type = int,
        default =32
        )
        parser.add_argument(
        '--single_optimizer',
        type = bool,
        default =False
        )
        parser.add_argument(
        '--new_method',
        type = bool,
        default =False
        )
        parser.add_argument(
        '--multiple_optimizers_metric',
        type = bool,
        default =False
        )
        parser.add_argument(
        '--multiple_optimizers_1_17',
        type = bool,
        default =False
        )
        parser.add_argument(
        '--multiple_optimizers_random',
        type = bool,
        default =False
        )
        FLAGS, unparsed = parser.parse_known_args()
        ex = FitNet() 
        tf.app.run(main=ex.main, argv = [sys.argv[0]] + unparsed)
