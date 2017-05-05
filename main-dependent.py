import tensorflow as tf
import random
from DataInput import DataInput
from DataInputTest import DataInputTest
from mentee import VGG16Mentee
from VGG16 import VGG16
from embed import Embed
import os
import pdb
dataset_path = "./"
train_labels_file = "dataset-train.txt"
test_labels_file = "dataset-test.txt"
epoch = 0
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_CHANNELS = 3
BATCH_SIZE = 40
NUM_ITERATIONS = 40000
LEARNING_RATE = 0.01
SUMMARY_LOG_DIR="./summary-log"
#ALPHA = 5.0
#BETA = 100.0
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
def get_variables_to_restore(variables_to_restore):

        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="conv1_1/weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="conv1_2/weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="conv2_1/weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="conv2_2/weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="conv3_1/weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="conv3_2/weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="conv3_3/weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="conv4_1/weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="conv4_2/weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="conv4_3/weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="conv5_1/weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="conv5_2/weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="conv5_3/weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="fc1/weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="fc2/weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="fc3/weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="conv1_1/biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="conv1_2/biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="conv2_1/biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="conv2_1/biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="conv2_2/biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="conv3_1/biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="conv3_2/biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="conv3_3/biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="conv4_1/biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="conv4_2/biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="conv4_3/biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="conv5_1/biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="conv5_2/biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="conv5_3/biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="fc1/biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="fc2/biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="fc3/biases"][0])

        return variables_to_restore


def evaluation(logits, labels):
	correct = tf.nn.in_top_k(logits, labels, 1)
	pred = tf.argmax(logits, 1)
	return tf.reduce_sum(tf.cast(correct, tf.int32))


def main():

	with tf.Graph().as_default():
		data_input_train = DataInput(dataset_path, train_labels_file, BATCH_SIZE)

		data_input_test = DataInputTest(dataset_path, test_labels_file,BATCH_SIZE)
		images_placeholder, labels_placeholder = placeholder_inputs(BATCH_SIZE)

		sess = tf.Session()
                #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
                
		vgg16_mentee = VGG16Mentee()
                vgg16_mentor = VGG16()
                               
		mentor_conv4, mentor_conv5, logits_mentor, softmax_temp_mentor = vgg16_mentor.build(images_placeholder)

                mentee_conv4, mentee_conv5, logits_mentee, softmax_temp_mentee = vgg16_mentee.build(images_placeholder)
                #embed = Embed()
                #embed_loss_1, embed_loss_2 = embed.build(images_placeholder, mentor_conv5, mentor_conv8, mentee_conv2, mentee_conv5)
                embed_loss_1 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(mentor_conv4, mentee_conv4))))
                embed_loss_2 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(mentor_conv5, mentee_conv5))))
                embed_loss = tf.add(embed_loss_1, embed_loss_2)

                temp_softmax_loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(softmax_temp_mentor, softmax_temp_mentee))))
                
                #temp_softmax_loss = tf.nn.l2_loss(softmax_temp_mentor- softmax_temp_mentee)/BATCH_SIZE
               # vgg16_mentee.build(images_placeholder)
                
                summary = tf.summary.merge_all()
                
                variables_to_restore = []
                variables_to_restore = get_variables_to_restore(variables_to_restore)
		saver = tf.train.Saver(variables_to_restore)
                
		summary_writer = tf.summary.FileWriter(SUMMARY_LOG_DIR, sess.graph)
		coord = tf.train.Coordinator()
                #var_list = vgg16.get_training_vars()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                global_step = tf.Variable(0, trainable = False)
                ## Adding Exponential Decay Learning Rate of 0.95 for every 10000 steps
                learning_rate = tf.train.exponential_decay(0.01,global_step,800,0.98,staircase = True)
                alpha = tf.train.exponential_decay(0.005,global_step,800,2,staircase = True)

                #alpha = tf.Variable(tf.cast(global_step, tf.float32))
                beta = tf.train.exponential_decay(0.15,global_step,800,0.95,staircase = True)
                gamma = tf.train.exponential_decay(0.01,global_step,800,0.95,staircase = True)
		mentee_logits_loss = vgg16_mentee.mentee_loss(labels_placeholder)
                
                total_mentee_loss = tf.add(tf.multiply(alpha, mentee_logits_loss) , tf.add(tf.multiply( beta, embed_loss), tf.multiply(gamma, temp_softmax_loss))) 
                #pdb.set_trace() 
                #total_mentee_loss = tf.add(tf.multiply( beta, embed_loss), tf.multiply(gamma, temp_softmax_loss)) 
                #total_mentee_loss = tf.add(mentee_logits_loss , tf.add(tf.multiply( tf.constant(ALPHA), embed_loss), tf.multiply(tf.constant(BETA), temp_softmax_loss))) 
		train_op = vgg16_mentee.training(total_mentee_loss, learning_rate,global_step)
		init = tf.initialize_all_variables()
		sess.run(init)
                saver.restore(sess, "./summary-log-2/model.ckpt-3060")
		eval_correct = evaluation(vgg16_mentee.fc2l, labels_placeholder)
		try:
			for i in range(NUM_ITERATIONS):

				feed_dict = fill_feed_dict(data_input_train, images_placeholder,
								labels_placeholder, sess)
				_, loss_value,embed_loss_print = sess.run([train_op,total_mentee_loss,embed_loss], feed_dict=feed_dict)
                                #pdb.set_trace()
				if i % 5 == 0:
					print ('Step %d: loss = %.2f' % (i, loss_value))
					summary_str = sess.run(summary, feed_dict=feed_dict)
					summary_writer.add_summary(summary_str, i)
					summary_writer.flush()

				if (i) % (30607//BATCH_SIZE) == 0 or (i) == NUM_ITERATIONS:
                                        global epoch
                                        epoch = epoch + 1
					checkpoint_file = os.path.join(SUMMARY_LOG_DIR, 'model.ckpt')
					saver.save(sess, checkpoint_file, global_step=i)  
                                        """
					print ("Training Data Eval:")
					do_eval(sess,
						eval_correct,
						vgg16_mentor.fc2,
						images_placeholder,
						labels_placeholder,
    						data_input_train)
                                        """
                                        print ("Testing Data Eval: Epoch =", epoch)
                                        do_eval(sess, 
                                                eval_correct,
                                                vgg16_mentee.fc2l,
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
