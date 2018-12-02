import tensorflow as tf
import numpy as np
import os 
import glob
import string
from PIL import Image, ImageFilter, ImageEnhance  
import tensorflow.contrib.slim as slim
import image_process


MODE = "test"
RESTORE = True
SAVE = True
TRAINING_STEPS = 20000 + 5
SIZE_TEST_SET = 100
SIZE_TRAIN_SET = 9900
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99


IMAGE_HEIGHT = 80
IMAGE_WIDTH = 200
MAX_CAPTCHA = 4
FOOT_TRAIN =  0
FOOT_VALIDATION = 0
OUTPUT_SIZE = 144
BATCH_SIZE = 128
TOTAL_NUM_EXAMPLE = 9900
SVAE_STEP = 500


MODEL_SAVE_PATH = "./model"
MODEL_NAME = "model.ckpt"
INPUT_DATA = "./process_data/"
SUMMARY_DIR ="./log"
INPUT_DATA_TRAIN = "./process_data/train/"
INPUT_DATA_TEST = "./process_data/test/"
LABEL_DIR = "./process_data/mappings.txt"
RESULT_DIR = './result/mappings.txt'

char_set = string.digits + string.ascii_uppercase
CHAR_SET_LEN = len(char_set)


def text2vec(text):
	text_len = len(text) 
	vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
	def char2pos(c):
		k = ord(c) - 48
		if k > 9:
			k = ord(c) - 55 
		return k

	for i, c in enumerate(text):
		idx = i * CHAR_SET_LEN + char2pos(c)
		vector[idx] = 1
	return vector 

def vec2text(vec):
	char_pos = vec.nonzero()[0]
	text=[]
	for i, c in enumerate(char_pos):
		char_at_pos = i 
		char_idx = c % CHAR_SET_LEN
		if char_idx < 10:
			char_code = char_idx + 48
		else:
			char_code = char_idx + 55
		text.append(chr(char_code))
	return "".join(text)

def get_next_batch_train(batch_size = 64):
	batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
	batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])
 
	global FOOT_TRAIN
	count = 0
	while True:
		if FOOT_TRAIN == SIZE_TRAIN_SET:
 			FOOT_TRAIN = 0
		index = FOOT_TRAIN
		img_name = image_dict['train'][index]
		image_file = INPUT_DATA_TRAIN + img_name
		image = Image.open(image_file)
		# image = image_process.FinalProcess(image)	
		image = np.array(image)
		text = label_dict[image_dict['train'][index].split('.')[0]]
		batch_x[count, :] = image.flatten() / 128 - 1
		batch_y[count, :] = text2vec(text)
		count += 1
		FOOT_TRAIN += 1
		if count == batch_size:
			break
	return batch_x, batch_y
 

def crack_captcha_cnn():
	x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

	conv_1 = slim.conv2d(x, 64, [3, 3], 1, padding = 'SAME', scope = 'conv1', activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer_conv2d())
	max_pool_1 = slim.max_pool2d(conv_1, [2, 2], [2, 2], padding = 'SAME')
	conv_2 = slim.conv2d(max_pool_1, 128, [3, 3], padding = 'SAME', scope = 'conv2', activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer_conv2d())
	max_pool_2 = slim.max_pool2d(conv_2, [2, 2], [2, 2], padding = 'SAME')
	conv_3 = slim.conv2d(max_pool_2, 256, [3, 3], padding = 'SAME', scope = 'conv3', activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer_conv2d())
	max_pool_3 = slim.max_pool2d(conv_3, [2, 2], [2, 2], padding = 'SAME')

	flatten = slim.flatten(max_pool_3)
	fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024, activation_fn = tf.nn.relu, scope = 'fc1')
	logits = slim.fully_connected(slim.dropout(fc1, keep_prob), OUTPUT_SIZE, activation_fn = None, scope = 'fc2')

	return logits

def create_images_lists():

    result = {}

    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]

    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        extension = 'jpg'
        file_list = []
        dir_name = os.path.basename(sub_dir)
        file_glob = os.path.join(INPUT_DATA, dir_name, "*." + extension)
        file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue

        images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            images.append(base_name)
        result[dir_name] = images
    
    return result

def create_labels_lists():
    file = open(LABEL_DIR, 'r')
    label_dict = {}
    for line in file:
        key, value = line.split(",")
        value = value[0: 4]
        label_dict[key] = value
    file.close()

    return label_dict


image_dict = create_images_lists()
label_dict = create_labels_lists()

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)


def train_crack_captcha_cnn():
	with tf.device("/cpu:0"):
		output = crack_captcha_cnn()
		global_step = tf.Variable(0, trainable = False)

		with tf.name_scope('loss'):
			loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = output, labels = Y))
			tf.summary.scalar('loss',loss)

		learning_rate = tf.train.exponential_decay(
			LEARNING_RATE_BASE,
			global_step,
			TOTAL_NUM_EXAMPLE / BATCH_SIZE,
			LEARNING_RATE_DECAY
			)

		optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = global_step)

		with tf.name_scope('accuracy'):
			with tf.name_scope('correct_pred'):
				predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])

				max_idx_p = tf.argmax(predict, 2)
				lable_predict = tf.reshape(Y,[-1, MAX_CAPTCHA, CHAR_SET_LEN])
				max_idx_l = tf.argmax(lable_predict, 2)
				correct_pred = tf.equal(max_idx_p, max_idx_l)
			with tf.name_scope('accuracy'):
				accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
			tf.summary.scalar('accuracy', accuracy)	

		merged = tf.summary.merge_all()
		with tf.Session() as sess:

			saver = tf.train.Saver()
			summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
			sess.run(tf.global_variables_initializer())

			if RESTORE:
				ckpt = tf.train.latest_checkpoint(MODEL_SAVE_PATH)
				if ckpt:
					saver.restore(sess, ckpt)
				else:
					print("No found checkpoint!")

			for i in range(TRAINING_STEPS):
				batch_x, batch_y = get_next_batch_train(BATCH_SIZE)
				summary, _ , loss_ , step = sess.run([merged, optimizer, loss, global_step], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})

				summary_writer.add_summary(summary,step) 

				print("After %d steps training ,the loss is %g"%(step, loss_))
				
				if step % SVAE_STEP == 0 and SAVE:
					saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step)
					print(" ")
					print('==================================================')
					print("After %d steps training, model save succeed" %(step))
					print('==================================================')
					print(" ")

			summary_writer.close()



# ---------------------------------------

def test_crack_captcha_cnn():

	file = open(RESULT_DIR,'w')
	output = crack_captcha_cnn()
	predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
	# error = []
	saver = tf.train.Saver()
	
	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)

		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print("Load successfully!")
		else:
			print("No found checkpoint!")

		
		correct_num = 0

		for i in range(SIZE_TEST_SET):
			img_name = image_dict['test'][i]
			image_file = os.path.join(INPUT_DATA_TEST, img_name)
			image = Image.open(image_file)
			image = image_process.FinalProcess(image)
			image = np.array(image)
			image = image.flatten() / 128 - 1 

			text_list = sess.run(predict, feed_dict = {X: [image], keep_prob: 1.0})
	 		# correct = label_dict[img_name.split('.')[0]]
			text = text_list[0].tolist()
			vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
			j = 0
			for n in text:
				vector[j * CHAR_SET_LEN + n] = 1
				j += 1

			prediction = vec2text(vector)
			# result = "False"
			# if correct == prediction:
			# 	correct_num += 1		
			# 	result = "True"
			# else:
			# 	result = "False"
			# 	error.append(i)
			prediction = vec2text(vector)
			# print("No.{}  正确: {}  预测: {}  结果： {}".format(str(i), correct, prediction, result))
			file.write("%0.4d" % i + "," + vec2text(vector) + '\n')
		
		# print(" ")
		# print('======================================')
		# print("The final accurancy in the test set is %g%%." %(correct_num / SIZE_TEST_SET * 100.))	
		# print('======================================')
		# print(" ")

		# print("The index of error is ", end = "")
		# print(error)
	
	file.close()


if __name__ == '__main__':
	if MODE == "train":
		train_crack_captcha_cnn()
	elif MODE == "test":
		test_crack_captcha_cnn()