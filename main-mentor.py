import tensorflow as tf
import random
from DataInput import DataInput
from DataInputTest import DataInputTest
from VGG16 import VGG16
import os
import time
import pdb
import numpy as np
dataset_path = "./"
train_labels_file = "dataset-train.txt"
test_labels_file = "dataset-test.txt"
epoch = 0
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_CHANNELS = 3
BATCH_SIZE =40
NUM_ITERATIONS = 5000
LEARNING_RATE = 0.0001

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

        variables_to_restore.append([v for v in tf.global_variables() if v.name == "fc3/weights:0"][0])
        variables_to_restore.append([v for v in tf.global_variables() if v.name == "fc3/biases:0"][0])

        return variables_to_restore

def prune_weights(weights_prune, sess,threshold = 0.01):
    
        sparse_weights = {}
        for v in weights_prune:
            
            value = sess.run(v)
        
            under_threshold = abs(value) < threshold 
            value[under_threshold] =0
            sess.run(v.assign(value))
            sparse_weights[v.name] = -under_threshold
            count = np.sum(under_threshold)
            
            print ("Non-zero count (Sparse): %s" % (value.size - count))
        return sparse_weights

def calculate_no_of_parameters(weights_prune, sess, threshold = 0.01):
    for v in weights_prune:
        value = sess.run(v)
        under_threshold = abs(value) < threshold

        count = np.sum(under_threshold)
        print("Non-zero count (Sparse): %s" % (value.size - count))

"""
def gen_sparse_dict(weights_prune, sess, threshold = 0.01):
        sparse_w = {}
        for v in weights_prune:
            target_array = np.transpose(sess.run(v))
            under_threshold = abs(target_array) < threshold
            target_array[under_threshold] =0
            values = target_array[target_array !=0]
            indices = np.transpose(np.nonzero(target_array))
            shape = list(target_array.shape)
            name1 = v.name.split(":")
            sparse_w[name1[0] + "_idx"] = tf.Variable(tf.constant(indices,dtype=tf.int32),
                name=name1[0]+"_idx")
            sparse_w[name1[0] + "_sparse"]=tf.Variable(tf.constant(values,dtype=tf.float32),
                            name=name1[0]+ "_sparse")
            sparse_w[name1[0]+"_shape"]=tf.Variable(tf.constant(shape,dtype=tf.int32),
                        name=name1[0]+"_shape")
        return sparse_w
"""
def apply_prune_on_grads(grads_and_vars, sess, dict_n):

    for k, v in dict_n.items():
        count =0
        for grad, var in grads_and_vars:
             
            if var.name == k:
                op = (var.name).split("/")
                if op[1] != 'biases:0':

                    n_obj = tf.cast(tf.constant(v), tf.float32)
                    n_obj = sess.run(n_obj)
                    grads_and_vars[count] = (tf.multiply(n_obj, grad), var)
            count = count+1
    return grads_and_vars

def main():

	with tf.Graph().as_default():
		data_input_train = DataInput(dataset_path, train_labels_file, BATCH_SIZE)
		data_input_test = DataInputTest(dataset_path, test_labels_file, BATCH_SIZE)
		images_placeholder, labels_placeholder = placeholder_inputs(BATCH_SIZE)

		vgg16 = VGG16()
		vgg16.build(images_placeholder)
                variables_to_restore = []
                variables_to_restore = get_variables_to_restore(variables_to_restore)


		summary = tf.summary.merge_all()
		saver = tf.train.Saver(variables_to_restore)
		sess = tf.Session()
		summary_writer = tf.summary.FileWriter(SUMMARY_LOG_DIR, sess.graph)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

		loss = vgg16.loss(labels_placeholder)
		train_op = vgg16.training(loss, LEARNING_RATE)
		init = tf.initialize_all_variables()

		sess.run(init)
                saver.restore(sess, "./summary-log-2/model.ckpt-3060")
                sparse_weights = prune_weights(variables_to_restore, sess)
                #gen_sparse_dict(variables_to_restore, sess)
                grads_and_vars = train_op.compute_gradients(loss)
                grads_and_vars =  apply_prune_on_grads(grads_and_vars,sess,  sparse_weights)
                train_step = train_op.apply_gradients(grads_and_vars)
                    
                """
                for var in tf.all_variables():
                    if sess.run(tf.is_variable_initialized(var)) == False:
                        sess.run(tf.initialize_variables([var]))
                
                final_saver = tf.train.Saver(sparse_w)
                
                final_saver.save(sess, "./model_ckpt_sparse_retraied")
                """
		eval_correct = evaluation(vgg16.fc3l, labels_placeholder)
		try:
			for i in range(NUM_ITERATIONS):
				feed_dict = fill_feed_dict(data_input_train, images_placeholder,
								labels_placeholder, sess)
				_, loss_value = sess.run([train_step, loss], feed_dict=feed_dict)

				if i % 5 == 0:
					print ('Step %d: loss = %.2f' % (i, loss_value))
					summary_str = sess.run(summary, feed_dict=feed_dict)
					summary_writer.add_summary(summary_str, i)
					summary_writer.flush()
                                        calculate_no_of_parameters(variables_to_restore, sess)

				if (i) % (30607//BATCH_SIZE) == 0 or (i) == NUM_ITERATIONS:
                                        global epoch 
                                        epoch = epoch + 1
					checkpoint_file = os.path.join(SUMMARY_LOG_DIR, 'model.ckpt')
					saver.save(sess, checkpoint_file, global_step=i)
					print ("Training Data Eval:")
					do_eval(sess,
						eval_correct,
						vgg16.fc3l,
						images_placeholder,
						labels_placeholder,
    						data_input_train)
					print ("Testing Data Eval: EPOCH->", epoch)
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
