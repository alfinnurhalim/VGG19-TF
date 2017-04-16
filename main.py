import tensorflow as tf
import random
from DataInput import DataInput
from DataInputTest import DataInputTest
from mentee import VGG16Mentee
from mentor import VGG16Mentor
from embed import Embed
import os
import pdb
from tensorflow.python import debug as tf_debug
dataset_path = "./"
train_labels_file = "dataset-train.txt"
test_labels_file = "dataset-test.txt"

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_CHANNELS = 3
BATCH_SIZE = 4
NUM_ITERATIONS = 100000
LEARNING_RATE = 0.001
ALPHA = 5.0
BETA = 100.0
SUMMARY_LOG_DIR="./summary-log"

def placeholder_inputs(batch_size):
	images_placeholder = tf.placeholder(tf.float32, 
								shape=(batch_size, IMAGE_HEIGHT, 
									   IMAGE_WIDTH, NUM_CHANNELS))
	labels_placeholder = tf.placeholder(tf.int32,
								shape=(batch_size))

	return images_placeholder, labels_placeholder

def fill_feed_dict(data_input, images_pl, labels_pl, sess):
	images_feed, labels_feed = sess.run([data_input.example_batch, data_input.label_batch])
	feed_dict = {
		images_pl: images_feed,
		labels_pl: labels_feed,
	}
	return feed_dict

def do_eval(sess,
			eval_correct,
			logits,
			images_placeholder,
			labels_placeholder,
			dataset):

	true_count =0
	steps_per_epoch = dataset.num_examples // BATCH_SIZE
	num_examples = steps_per_epoch * BATCH_SIZE

	for step in xrange(steps_per_epoch):
		feed_dict = fill_feed_dict(dataset, images_placeholder,
									labels_placeholder,sess)
		count = sess.run(eval_correct, feed_dict=feed_dict)
		true_count = true_count + count

	precision = float(true_count) / num_examples
	print ('  Num examples: %d, Num correct: %d, Precision @ 1: %0.04f' %
			(num_examples, true_count, precision))

def evaluation(logits, labels):
	correct = tf.nn.in_top_k(logits, labels, 1)
	pred = tf.argmax(logits, 1)

	return tf.reduce_sum(tf.cast(correct, tf.int32))

def get_variables_to_restore(variables_to_restore):

        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv1_1/mentor_weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv2_1/mentor_weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv3_1/mentor_weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv4_1/mentor_weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv1_1/mentor_biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv2_1/mentor_biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv3_1/mentor_biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv4_1/mentor_biases"][0])

        return variables_to_restore

def main():

	with tf.Graph().as_default():
		data_input_train = DataInput(dataset_path, train_labels_file, BATCH_SIZE)

		data_input_test = DataInputTest(dataset_path, test_labels_file,BATCH_SIZE)
		images_placeholder, labels_placeholder = placeholder_inputs(BATCH_SIZE)
                
		sess = tf.Session()
                #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
                
		vgg16_mentor = VGG16Mentor()
                vgg16_mentee = VGG16Mentee()
		mentor_conv3, mentor_conv5, logits_mentor, softmax_temp_mentor = vgg16_mentor.build(images_placeholder)

                mentee_conv1, mentee_conv2, logits_mentee, softmax_temp_mentee = vgg16_mentee.build(images_placeholder)

                embed = Embed()
                embed_loss_1, embed_loss_2 = embed.build(images_placeholder, mentor_conv3, mentor_conv5, mentee_conv1, mentee_conv2)

                embed_loss = tf.add(embed_loss_1, embed_loss_2)

                temp_softmax_loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(softmax_temp_mentor, softmax_temp_mentee))))
                
                vgg16_mentee = VGG16Mentee()
                vgg16_mentee.build(images_placeholder)
		summary = tf.summary.merge_all()
                variables_to_restore = []
                variables_to_restore = get_variables_to_restore(variables_to_restore)
		saver = tf.train.Saver(variables_to_restore)
		summary_writer = tf.summary.FileWriter(SUMMARY_LOG_DIR, sess.graph)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		mentee_logits_loss = vgg16_mentee.mentee_loss(labels_placeholder)
                total_mentee_loss = tf.add(mentee_logits_loss , tf.add(tf.multiply( tf.constant(ALPHA), embed_loss), tf.multiply(tf.constant(BETA), temp_softmax_loss))) 
		train_op = vgg16_mentee.training(total_mentee_loss, LEARNING_RATE)
		init = tf.initialize_all_variables()
		sess.run(init)
                saver.restore(sess, "./summary-log/model.ckpt-79")
		eval_correct = evaluation(vgg16_mentee.fc1, labels_placeholder)
		try:
			for i in range(NUM_ITERATIONS):
				feed_dict = fill_feed_dict(data_input_train, images_placeholder,
								labels_placeholder, sess)

				_, loss_value = sess.run([train_op, total_mentee_loss], feed_dict=feed_dict)
				if i % 5 == 0:
					print ('Step %d: loss = %.2f' % (i, loss_value))

					summary_str = sess.run(summary, feed_dict=feed_dict)
					summary_writer.add_summary(summary_str, i)
					summary_writer.flush()

				if (i + 1) % 20 == 0 or (i + 1) == NUM_ITERATIONS:
					checkpoint_file = os.path.join(SUMMARY_LOG_DIR, 'model.ckpt')
					saver.save(sess, checkpoint_file, global_step=i)
					print ("Training Data Eval:")
					do_eval(sess,
						eval_correct,
						vgg16_mentee.fc1,
						images_placeholder,
						labels_placeholder,
    						data_input_train)
                                        print ("Testing Data Eval:")
                                        do_eval(sess, 
                                                eval_correct,
                                                vgg16_mentee.fc1,
                                                images_placeholder,
                                                labels_placeholder,
                                                data_input_test)
                            
			coord.request_stop()
			coord.join(threads)
		except Exception as e:
			print(e) 
	sess.close()

if __name__ == '__main__':
	main()
