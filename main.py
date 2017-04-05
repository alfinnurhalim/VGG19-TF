import tensorflow as tf
import random
from DataInput import DataInput
from DataInputTest import DataInputTest
from VGG16 import VGG16
import os
import time
import pdb
dataset_path = "./"
train_labels_file = "dataset_caltech256_train.txt"
test_labels_file = "dataset_caltech256_test.txt"

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_CHANNELS = 3
BATCH_SIZE =16
NUM_ITERATIONS = 5000
LEARNING_RATE = 0.0001
SUMMARY_LOG_DIR="./summary-log-2"


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

	true_count = 0
	steps_per_epoch = dataset.num_examples // BATCH_SIZE
	num_examples = steps_per_epoch * BATCH_SIZE
	for step in xrange(steps_per_epoch):
#                time.sleep(0.1)
                print("step value",step)
		feed_dict = fill_feed_dict(dataset, images_placeholder,
									labels_placeholder,sess)

#                pdb.set_trace()
		count = sess.run(eval_correct, feed_dict=feed_dict)
		true_count = true_count + count
	precision = float(true_count) / num_examples
        #pdb.set_trace()
	print ('  Num examples: %d, Num correct: %d, Precision @ 1: %0.04f' %
			(num_examples, true_count, precision))

def evaluation(logits, labels):
	correct = tf.nn.in_top_k(logits, labels, 1)
	pred = tf.argmax(logits, 1)

	return tf.reduce_sum(tf.cast(correct, tf.int32))

def get_variables_to_restore(variables_to_restore):

        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="conv1_1/weights"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "conv1_1/biases:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "conv1_2/weights:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "conv1_2/biases:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "conv2_1/weights:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "conv2_1/biases:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "conv2_2/weights:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "conv2_2/biases:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "conv3_1/weights:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "conv3_1/biases:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "conv3_2/weights:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "conv3_2/biases:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "conv3_3/weights:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "conv3_3/biases:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "conv4_1/weights:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "conv4_1/biases:0"][0])

        variables_to_restore.append([v for v in tf.global_variables() if v.name == "conv4_2/weights:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "conv4_2/biases:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "conv4_3/weights:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "conv4_3/biases:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "conv5_1/weights:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "conv5_1/biases:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "conv5_2/weights:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "conv5_2/biases:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "conv5_3/weights:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "conv5_3/biases:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "fc1/weights:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "fc1/biases:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "fc2/weights:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "fc2/biases:0"][0])

def main():

	with tf.Graph().as_default():
		data_input_train = DataInput(dataset_path, train_labels_file, BATCH_SIZE)
		data_input_test = DataInputTest(dataset_path, test_labels_file, BATCH_SIZE)
		images_placeholder, labels_placeholder = placeholder_inputs(BATCH_SIZE)

		vgg16 = VGG16()
		vgg16.build(images_placeholder)
                variables_to_restore = []
                get_variables_to_restore(variables_to_restore)
		summary = tf.summary.merge_all()
		saver = tf.train.Saver(variables_to_restore)
		sess = tf.Session()
		summary_writer = tf.summary.FileWriter(SUMMARY_LOG_DIR, sess.graph)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

		loss = vgg16.loss(labels_placeholder)
                training_vars = vgg16.get_training_vars()
		train_op = vgg16.training(loss, LEARNING_RATE, training_vars)

		#init = tf.initialize_all_variables()
                init = tf.global_variables_initializer()
		sess.run(init)
                saver.restore(sess, "./summary-log/model.ckpt-4999")
		eval_correct = evaluation(vgg16.fc3l, labels_placeholder)
		try:
			for i in range(NUM_ITERATIONS):
				feed_dict = fill_feed_dict(data_input_train, images_placeholder,
								labels_placeholder, sess)
	                        #pdb.set_trace()		    
				_, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
				if i % 5 == 0:
					print ('Step %d: loss = %.2f' % (i, loss_value))
					summary_str = sess.run(summary, feed_dict=feed_dict)
					summary_writer.add_summary(summary_str, i)
					summary_writer.flush()

				if (i + 1) % 20 == 0 or (i + 1) == NUM_ITERATIONS:
					checkpoint_file = os.path.join(SUMMARY_LOG_DIR, 'model.ckpt')
					saver.save(sess, checkpoint_file, global_step=i)
					print ("Testing Data Eval:")
                                        
					do_eval(sess,
						eval_correct,
						vgg16.fc3l,
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
