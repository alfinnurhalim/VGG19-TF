import tensorflow as tf
import random
from DataInput import DataInput

dataset_path = "./"
train_labels_file = "dataset.txt"

test_set_size = 1829
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_CHANNELS = 3

BATCH_SIZE = 5

def main():

	data_input = DataInput(dataset_path, train_labels_file, test_set_size, BATCH_SIZE, BATCH_SIZE)

	with tf.Session() as sess:

		sess.run(tf.initialize_all_variables())

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		for i in range(20):
			print sess.run(data_input.train_label_batch)

		for i in range(20):
			print sess.run(data_input.test_label_batch)	

	coord.request_stop()
	coord.join(threads)
	sess.close()

if __name__ == '__main__':
	main()