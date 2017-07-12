import tensorflow as tf
import random
import os
from DataInput import DataInput
#from VGG16 import VGG16
from VGG13 import VGG13
import pdb
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

dataset_path = "./"
train_labels_file = "dataset.txt"

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_CHANNELS = 3
BATCH_SIZE = 40
NUM_ITERATIONS = 2000
LEARNING_RATE = 0.001
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

	true_count = 0
	steps_per_epoch = dataset.num_examples // BATCH_SIZE
	num_examples = steps_per_epoch * BATCH_SIZE

	for step in xrange(steps_per_epoch):
		#feed_dict = fill_feed_dict(dataset, images_placeholder,	labels_placeholder)
		feed_dict = fill_feed_dict(dataset, images_placeholder,	labels_placeholder,sess)
		count = sess.run(eval_correct, feed_dict=feed_dict)
		true_count = true_count + count

	precision = float(true_count) / num_examples
	print ('  Num examples: %d, Num correct: %d, Precision @ 1: %0.04f' %
			(num_examples, true_count, precision))

def evaluation(logits, labels):
	correct = tf.nn.in_top_k(logits, labels, 1)
	pred = tf.argmax(logits, 1)

	return tf.reduce_sum(tf.cast(correct, tf.int32))


def main():

	with tf.Graph().as_default():
		data_input = DataInput(dataset_path, train_labels_file, BATCH_SIZE)
		images_placeholder, labels_placeholder = placeholder_inputs(BATCH_SIZE)

		vgg13 = VGG13()
		vgg13.build(images_placeholder)

		summary = tf.summary.merge_all()
		saver = tf.train.Saver()
		sess = tf.Session()
		summary_writer = tf.summary.FileWriter(SUMMARY_LOG_DIR, sess.graph)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

		loss = vgg13.loss(labels_placeholder)
		train_op = vgg13.training(loss, LEARNING_RATE)

                # deprecated
		#init = tf.initialize_all_variables()
		init = tf.global_variables_initializer()
		sess.run(init)
		eval_correct = evaluation(vgg13.fc3l, labels_placeholder)
		try:
			for i in range(NUM_ITERATIONS):
				feed_dict = fill_feed_dict(data_input, images_placeholder,
								labels_placeholder, sess)
				
				_, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
				if i % 5 == 0:
					print ('Step %d: loss = %.2f' % (i, loss_value))

					summary_str = sess.run(summary, feed_dict=feed_dict)
					summary_writer.add_summary(summary_str, i)
					summary_writer.flush()

				if (i + 1) % 100 == 0 or (i + 1) == NUM_ITERATIONS:
				#if (i + 1) % 10 == 0 or (i + 1) == NUM_ITERATIONS:
					checkpoint_file = os.path.join(SUMMARY_LOG_DIR, 'model.ckpt')
					#saver.save(sess, checkpoint_file, global_step=step)
					saver.save(sess, checkpoint_file, global_step=i)
					print ("Training Data Eval:")
					do_eval(sess,
						eval_correct,
						vgg13.fc3l,
						images_placeholder,
						labels_placeholder,
						#dataset)
						data_input)

			coord.request_stop()
			coord.join(threads)
		except Exception as e:
			print(e) 
	sess.close()

if __name__ == '__main__':
	main()
