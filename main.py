import tensorflow as tf 
import random
from DataInput import DataInput
from DataInputTest import DataInputTest
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
train_labels_file = "train_map.txt"
test_labels_file = "test_map.txt"
epoch = 0
IMAGE_HEIGHT = 32
IMAGE_WIDTH =32
NUM_CHANNELS = 3
NUM_EPOCHS_PER_DECAY = 1.0
NUM_ITERATIONS = 5000000
FINAL_LEARNING_RATE = 0.0000002
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
LEARNING_RATE_DECAY_FACTOR = 0.9809
lamda = 4
SUMMARY_LOG_DIR="./summary-log"

def placeholder_inputs(batch_size):
	images_placeholder = tf.placeholder(tf.float32, 
								shape=(batch_size, IMAGE_HEIGHT, 
									   IMAGE_WIDTH, NUM_CHANNELS))
	labels_placeholder = tf.placeholder(tf.int32,
								shape=(batch_size))

	return images_placeholder, labels_placeholder

def fill_feed_dict(data_input, images_pl, labels_pl,sess, phase_train, mode):
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

def do_eval(sess,
			eval_correct,
			logits,
			images_placeholder,
			labels_placeholder,
			dataset, phase_train, mode):

	true_count = 0
	steps_per_epoch = dataset.num_examples //FLAGS.batch_size 
	num_examples = steps_per_epoch * FLAGS.batch_size
	for step in xrange(steps_per_epoch):
		feed_dict = fill_feed_dict(dataset, images_placeholder,
									labels_placeholder,sess, phase_train, mode)

		count = sess.run(eval_correct, feed_dict=feed_dict)
		true_count = true_count + count
	precision = float(true_count) / num_examples
	print ('  Num examples: %d, Num correct: %d, Precision @ 1: %0.04f' %
			(num_examples, true_count, precision))

def evaluation(logits, labels):
	correct = tf.nn.in_top_k(logits, labels, 1)
	return tf.reduce_sum(tf.cast(correct, tf.int32))

def get_variables_to_restore(variables_to_restore):

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

        return variables_to_restore

def get_variables_to_restore_KD(variables_to_restore):

        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="teacher_conv1_1/weights"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "teacher_conv1_1/biases:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "teacher_conv2_1/weights:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "teacher_conv2_1/biases:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "teacher_conv3_1/weights:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "teacher_conv3_1/biases:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "teacher_fc1/weights:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "teacher_fc1/biases:0"][0])
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
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv12_1/weights:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv12_1/biases:0"][0])

        variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv13_1/weights:0"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="student_conv13_1/weights"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv14_1/weights:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv14_1/biases:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv15_1/weights:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv15_1/biases:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv16_1/weights:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv16_1/biases:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv17_1/weights:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_conv17_1/biases:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_fc1/weights:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_fc1/biases:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_fc2/weights:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "student_fc2/biases:0"][0])

        return variables_to_restore
def main(_):

	with tf.Graph().as_default():
		data_input_train = DataInput(dataset_path, train_labels_file, FLAGS.batch_size)
		data_input_test = DataInputTest(dataset_path, test_labels_file,FLAGS.batch_size)
		images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

		summary = tf.summary.merge_all()
		sess = tf.Session()
		summary_writer = tf.summary.FileWriter(SUMMARY_LOG_DIR, sess.graph)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		teacher = Teacher()
                student = Student()
                keep_prob = tf.placeholder(tf.float32)
                phase_train = tf.placeholder(tf.bool, name = 'phase_train')
                global_step = tf.Variable(0, name='global_step', trainable=False)
                if FLAGS.teacher:
                    print("Teacher")
                    trainable = True
                    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
                    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
		    teacher.build(images_placeholder, trainable, phase_train)
                    loss = teacher.loss(labels_placeholder)

                    lr = tf.train.exponential_decay(FLAGS.learning_rate,global_step, decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)
                    train_op = teacher.training(loss, lr, global_step)
                    softmax = teacher.fc3
		    init = tf.initialize_all_variables()
                    sess.run(init)
                    saver = tf.train.Saver()

                elif (FLAGS.student and FLAGS.HT):
                    print("Student with Hind based approach")
                    trainable = False
                    teacher_second_layer_loss, _ = teacher.build(images_placeholder, trainable, phase_train)
                    student_eleventh_layer_loss, _ = student.build(images_placeholder)
                    embed = Embed()
                    loss = embed.build(teacher_second_layer_loss, student_eleventh_layer_loss, 'HT')
                    variables_to_restore = []
                    variables_to_restore = get_variables_to_restore(variables_to_restore)
                    saver = tf.train.Saver(variables_to_restore)
                    train_op = student.training(loss, FLAGS.learning_rate) 
                    softmax = student.fc2
		    init = tf.initialize_all_variables()
		    sess.run(init)
                    saver.restore(sess, FLAGS.teacher_weights_filename)

                elif (FLAGS.student and FLAGS.KD):
                    print("Student with Knowledge Distillation Approach")
                    trainable = False
                    _, teacher_softmax_layer_loss = teacher.build(images_placeholder, trainable, phase_train)
                    _,student_softmax_layer_loss = student.build(images_placeholder)
                    embed = Embed()
                    variables_to_restore = []
                    variables_to_restore = get_variables_to_restore_KD(variables_to_restore)
                    softmax_loss = embed.build(teacher_softmax_layer_loss, student_softmax_layer_loss, 'KD')
                    saver = tf.train.Saver(variables_to_restore)
                    #softmax_loss = tf.reduce_mean(tf.square(tf.subtract(student_softmax_layer_loss, teacher_softmax_layer_loss))) 
                    loss = student.loss(labels_placeholder)
                    global_step = tf.Variable(0, trainable=False)
                    lamda = tf.train.exponential_decay(lamda, global_step, 10000, 1.0, staircase = True)
                    total_loss = loss + lamda*softmax_loss 
                    train_op = student.training(total_loss, FLAGS.learning_rate)
                    softmax = student.fc2
		    init = tf.initialize_all_variables()
		    sess.run(init)
                    saver.restore(sess, FLAGS.HT_filename)

                elif FLAGS.student:
                    print("Independent student")
                    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
                    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
                    learning_rate_decay_factor = (FINAL_LEARNING_RATE/FLAGS.learning_rate)^(1/NUM_EPOCHS_PER_DECAY)
                    student.build(images_placeholder)
                    loss = student.loss(labels_placeholder)
                    lr = tf.train.exponential_decay(FLAGS.learning_rate,global_step, decay_steps,learning_rate_decay_factor,staircase=True)
                    train_op = student.training(loss, lr)
                    softmax = student.fc2
                    init = tf.initialize_all_variables()
                    sess.run(init)
                    saver = tf.train.Saver()
		eval_correct = evaluation(softmax, labels_placeholder)
		try:
			for i in range(NUM_ITERATIONS):
				feed_dict = fill_feed_dict(data_input_train, images_placeholder,
								labels_placeholder, sess, phase_train, 'Train')
                                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)                
				if i % 5 == 0:
					print ('Step %d: loss = %.5f' % (i, loss_value))
					summary_str = sess.run(summary, feed_dict=feed_dict)
					summary_writer.add_summary(summary_str, i)
					summary_writer.flush()

				if (i) % (50000//FLAGS.batch_size) == 0 or (i) == NUM_ITERATIONS:
                                        global epoch 
                                        epoch = epoch + 1
					checkpoint_file = os.path.join(SUMMARY_LOG_DIR, 'model.ckpt')
                                        if FLAGS.teacher:
					    saver.save(sess, FLAGS.teacher_weights_filename)
                                        
                                        elif FLAGS.student and FLAGS.HT:
                                            saver = tf.train.Saver()
                                            saver.save(sess, FLAGS.HT_filename)

                                        elif FLAGS.student and FLAGS.KD:
                                            saver.save(sess, FLAGS.KD_filename)

                                        elif FLAGS.student:
                                            saver.save(sess, FLAGS.student_filename)
					print ("Training Data Eval:")
					do_eval(sess,
						eval_correct,
						softmax,
						images_placeholder,
						labels_placeholder,
    						data_input_train,
                                                phase_train,
                                                'Test')
					print ("Testing Data Eval: EPOCH->", epoch)
					do_eval(sess,
						eval_correct,
						softmax,
						images_placeholder,
						labels_placeholder,
    						data_input_test,
                                                phase_train,
                                                'Test'
                                                )
                                        
                            
			coord.request_stop()
			coord.join(threads)
		except Exception as e:
			print(e) 
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
        '--HT_filename',
        type = str,
        default = "./summary-log/HT_filename")

        parser.add_argument(
        '--KD_filename',
        type = str,
        default = "./summary-log/KD_filename")

        parser.add_argument(
        '--teacher_weights_filename',
        type = str,
        default = "./summary-log/teacher_weights_filename"
        )
        parser.add_argument(
        '--student_filename',
        type = str,
        default = "./summary-log/student_filename"
        )

        parser.add_argument(
        '--learning_rate',
        type = float,
        default = 0.0001
        )

        parser.add_argument(
        '--batch_size',
        type = int,
        default = 50
        
        )
        FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv = [sys.argv[0]] + unparsed)
