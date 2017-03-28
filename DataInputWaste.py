
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random

NUM_CHANNELS = 3
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

class DataInput(object):

	def __init__(self, dataset_path, train_labels_file, test_set_size, train_batch_size, test_batch_size):
		self.dataset_path = dataset_path
		self.train_labels_file = train_labels_file
		self.test_set_size = test_set_size

		self.train_file_paths, self.train_labels = self.read_label_file(self.dataset_path + self.train_labels_file)

		self.train_file_paths = [dataset_path + fp for fp in self.train_file_paths]

		self.all_filepaths = self.train_file_paths
		self.all_labels = self.train_labels

		self.all_images = ops.convert_to_tensor(self.all_filepaths, dtype=dtypes.string)
		self.all_labels = ops.convert_to_tensor(self.all_labels, dtype=dtypes.int32)

		self.partition()
		self.build_queue()
		self.group_to_batch(train_batch_size, test_batch_size)

	def encode_label(self, label):
		return int(label)

	def read_label_file(self, file):
		f = open(file, "r")
		filepaths = []
		labels = []
		for line in f:
			filepath, label = line.split(",")
			filepaths.append(filepath)
			labels.append(self.encode_label(label))
		return filepaths, labels

	def partition(self):
		self.partitions = [0] * len(self.all_filepaths)
		self.partitions[:self.test_set_size] = [1] * self.test_set_size
		random.shuffle(self.partitions)

		self.train_images, self.test_images = tf.dynamic_partition(self.all_images, self.partitions, 2)
		self.train_labels, self.test_labels = tf.dynamic_partition(self.all_labels, self.partitions, 2)

	def build_queue(self): 
		self.train_input_queue = tf.train.slice_input_producer([self.train_images, self.train_labels], shuffle=False)
		self.test_input_queue = tf.train.slice_input_producer([self.test_images, self.test_labels], shuffle=False)

		file_content = tf.read_file(self.train_input_queue[0])
		self.train_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
		self.train_label = self.train_input_queue[1]

		file_content = tf.read_file(self.test_input_queue[0])
		self.test_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
		self.test_label = self.test_input_queue[1]

	def group_to_batch(self, train_batch_size, test_batch_size):
		self.train_image.set_shape([IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS])
		self.test_image.set_shape([IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS])

		self.train_image_batch, self.train_label_batch = tf.train.batch(
														[self.train_image, self.train_label],
														batch_size=train_batch_size,
														allow_smaller_final_batch=True)
		self.test_image_batch, self.test_label_batch = tf.train.batch(
														[self.test_image, self.test_label],
														batch_size=test_batch_size,
														allow_smaller_final_batch=True)