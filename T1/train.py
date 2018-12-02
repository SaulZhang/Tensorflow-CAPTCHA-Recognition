import tensorflow as tf
import numpy as np
import random 
import os 
import glob
from PIL import Image 
import tensorflow.contrib.slim as slim
import image_process


MODE = "test"
RESTORE = True
SAVE = True
TRAINING_STEPS = 5000 + 5
SIZE_TEST_SET = 100
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99


MODEL_SAVE_PATH = './model'
MODEL_NAME = 'model.ckpt'
SUMMARY_DIR ='./log'
INPUT_DATA_TRAIN = './process_data/train/'
INPUT_DATA_TEST = './process_data/test/'
INPUT_DATA = "./process_data/"
LABEL_DIR = "./process_data/mappings.txt"
RESULT_DIR = './result/mappings.txt'


SAVE_STEP = 500
BATCH_SIZE = 128
OUTPUT_NODE = 13
IMAGE_HEIGHT = 48
IMAGE_WIDTH = 36


FOOT_SUB =  0
Total_FOOT_SUB = 6544
FOOT_MUL = 0
Total_FOOT_MUL = 6642
FOOT_ADD = 0
Total_FOOT_ADD = 6614


FOOT_0 = 0
Total_FOOT_0 = 3056
FOOT_1 = 0
Total_FOOT_1 = 17658
FOOT_2 = 0
Total_FOOT_2 = 4485
FOOT_3 = 0
Total_FOOT_3 = 2957
FOOT_4 = 0
Total_FOOT_4 = 3055
FOOT_5 = 0
Total_FOOT_5 = 2926
FOOT_6 = 0
Total_FOOT_6 = 2915
FOOT_7 = 0
Total_FOOT_7 = 2976
FOOT_8 = 0
Total_FOOT_8 = 3016
FOOT_9 = 0
Total_FOOT_9 = 2848

TOTAL_NUM_EXAMPLE = 65692

number = ['0','1','2','3','4','5','6','7','8','9']
char = ['+','-','*']
char_set = number + char
CHAR_SET_LEN = len(char_set) 

def text2vec(char):
	pos = 0
	vector = np.zeros(OUTPUT_NODE)
	if  ord(char) == 43:
		pos = ord(char) - 33
	if 	ord(char) == 45:
		pos = ord(char) - 34
	if  ord(char) == 42:
		pos = ord(char) - 30
	if  48 <= ord(char) and ord(char) <= 57:
		pos = ord(char) - 48
	vector[pos] = 1
	return vector 

def vec2text(vec):
	char_pos = vec.nonzero()[0]
	if 0 <= char_pos and char_pos <= 9:
		char_code = char_pos + ord('0')
	if char_pos == 10:
		char_code = ord('+')
	if char_pos == 11:
		char_code = ord('-')
	if char_pos == 12:
		char_code = ord('*')
	return char_code

def get_next_batch_train(batch_size=128):
	batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
	batch_y = np.zeros([batch_size, OUTPUT_NODE])

	global FOOT_ADD, Total_FOOT_ADD, FOOT_SUB, Total_FOOT_SUB, FOOT_MUL, Total_FOOT_MUL
	global FOOT_0, Total_FOOT_0, FOOT_1, Total_FOOT_1, FOOT_2, Total_FOOT_2, FOOT_3, Total_FOOT_3
	global FOOT_4, Total_FOOT_4, FOOT_5, Total_FOOT_5, FOOT_6, Total_FOOT_6, FOOT_7, Total_FOOT_7
	global FOOT_8, Total_FOOT_8, FOOT_9, Total_FOOT_9

	Totalcount = 1
	while True:
		count = Totalcount - 1
		rand = random.randint(1, 13)

		if rand == 1:
			if FOOT_ADD == Total_FOOT_ADD:
				FOOT_ADD = 0
			path = os.path.join(INPUT_DATA_TRAIN, '+', train_image_dict['+'][FOOT_ADD])
			FOOT_ADD += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 255 
			batch_y[count, :] = text2vec('+')
		elif rand == 2:
			if FOOT_SUB == Total_FOOT_SUB:
				FOOT_SUB = 0
			path = os.path.join(INPUT_DATA_TRAIN, '-', train_image_dict['-'][FOOT_SUB])
			FOOT_SUB += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 255
			batch_y[count, :] = text2vec('-')
		elif rand == 3:
			if FOOT_MUL == Total_FOOT_MUL:
				FOOT_MUL = 0
			path = os.path.join(INPUT_DATA_TRAIN, '×', train_image_dict['×'][FOOT_MUL])
			FOOT_MUL += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 255
			batch_y[count, :] = text2vec('*')
		elif rand == 4:
			if FOOT_0 == Total_FOOT_0:
				FOOT_0 = 0
			path = os.path.join(INPUT_DATA_TRAIN, '0', train_image_dict['0'][FOOT_0])
			FOOT_0 += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 255
			batch_y[count, :] = text2vec('0')
		elif rand == 5:
			if FOOT_1 == Total_FOOT_1:
				FOOT_1 = 0				
			path = os.path.join(INPUT_DATA_TRAIN, '1', train_image_dict['1'][FOOT_1])
			FOOT_1 += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 255
			batch_y[count, :] = text2vec('1')
		elif rand == 6:
			if FOOT_2 == Total_FOOT_2:
				FOOT_2 = 0
			path = os.path.join(INPUT_DATA_TRAIN, '2', train_image_dict['2'][FOOT_2])
			FOOT_2 += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 255 
			batch_y[count, :] = text2vec('2')
		elif rand == 7:
			if FOOT_3 == Total_FOOT_3:
				FOOT_3 = 0
			path = os.path.join(INPUT_DATA_TRAIN, '3', train_image_dict['3'][FOOT_3])
			FOOT_3 += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 255 
			batch_y[count, :] = text2vec('3')
		elif rand == 8:
			if FOOT_4 == Total_FOOT_4:
				FOOT_4 = 0
			path = os.path.join(INPUT_DATA_TRAIN, '4', train_image_dict['4'][FOOT_4])
			FOOT_4 += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 255 
			batch_y[count, :] = text2vec('4')
		elif rand == 9:
			if FOOT_5 == Total_FOOT_5:
				FOOT_5 = 0
			path = os.path.join(INPUT_DATA_TRAIN, '5', train_image_dict['5'][FOOT_5])
			FOOT_5 += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 255
			batch_y[count, :] = text2vec('5')
		elif rand == 10:
			if FOOT_6 == Total_FOOT_6:
				FOOT_6 = 0				
			path = os.path.join(INPUT_DATA_TRAIN, '6', train_image_dict['6'][FOOT_6])
			FOOT_6 += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 255 
			batch_y[count, :] = text2vec('6')
		elif rand == 11:
			if FOOT_7 == Total_FOOT_7:
				FOOT_7 = 0			
			path = os.path.join(INPUT_DATA_TRAIN, '7', train_image_dict['7'][FOOT_7])
			FOOT_7 += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 255  
			batch_y[count, :] = text2vec('7')
		elif rand == 12:
			if FOOT_8 == Total_FOOT_8:
				FOOT_8 = 0
			path = os.path.join(INPUT_DATA_TRAIN, '8', train_image_dict['8'][FOOT_8])
			FOOT_8 += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 255 
			batch_y[count, :] = text2vec('8')
		elif rand == 13:
			if FOOT_9 == Total_FOOT_9:
				FOOT_9 = 0
			path = os.path.join(INPUT_DATA_TRAIN, '9', train_image_dict['9'][FOOT_9])
			FOOT_9 += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 255
			batch_y[count, :] = text2vec('9')

		Totalcount += 1
		if Totalcount == batch_size + 1:
			break		
	return batch_x, batch_y
 

def crack_captcha_cnn():
	x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
	
	conv_1 = slim.conv2d(x, 64, [3, 3], 1, padding ='SAME', scope = 'conv1', activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer_conv2d())
	max_pool_1 = slim.max_pool2d(conv_1, [2, 2], [2, 2], padding ='SAME')
	conv_2 = slim.conv2d(max_pool_1, 128, [3, 3], padding = 'SAME', scope = 'conv2', activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer_conv2d())
	max_pool_2 = slim.max_pool2d(conv_2, [2, 2], [2, 2], padding ='SAME')
	conv_3 = slim.conv2d(max_pool_2, 256, [3, 3], padding = 'SAME', scope = 'conv3', activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer_conv2d())
	max_pool_3 = slim.max_pool2d(conv_3, [2, 2], [2, 2], padding ='SAME')

	flatten = slim.flatten(max_pool_3)
	fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024, activation_fn = tf.nn.relu, scope = 'fc1')
	logits = slim.fully_connected(slim.dropout(fc1, keep_prob), OUTPUT_NODE, activation_fn = None, scope = 'fc2')

	return logits

def create_train_images_lists():

    result = {}

    sub_dirs = [x[0] for x in os.walk(INPUT_DATA_TRAIN)]

    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        extension = 'jpg'
        file_list = []
        dir_name = os.path.basename(sub_dir)
        file_glob = os.path.join(INPUT_DATA_TRAIN, dir_name, "*." + extension)
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
        value = value[0: len(value) - 1]
        label_dict[key] = value
    file.close()

    return label_dict


train_image_dict = create_train_images_lists()
label_dict = create_labels_lists()

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, OUTPUT_NODE])
keep_prob = tf.placeholder(tf.float32)


def train_crack_captcha_cnn():
	output = crack_captcha_cnn()
	global_step = tf.Variable(0,trainable=False)

	with tf.name_scope('cross_entropy_mean'):
		cross_entropy_mean = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = output, labels = Y))
		tf.summary.scalar('cross_entropy',cross_entropy_mean)
	
	learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_BASE,
		global_step,
		TOTAL_NUM_EXAMPLE / BATCH_SIZE,
		LEARNING_RATE_DECAY
		)

	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_mean, global_step = global_step)

	merged = tf.summary.merge_all()
	saver = tf.train.Saver()
	with tf.Session() as sess:
		summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
		tf.global_variables_initializer().run()

		if RESTORE:
			ckpt = tf.train.latest_checkpoint(MODEL_SAVE_PATH)
			if ckpt:
				saver.restore(sess, ckpt)
			else:
				print("No found checkpoint!")
 
		for i in range(TRAINING_STEPS):
			batch_x, batch_y = get_next_batch_train(BATCH_SIZE)
			summary, _ , loss, step = sess.run([merged, optimizer, cross_entropy_mean, global_step], feed_dict = {X: batch_x, Y: batch_y, keep_prob: 0.8})
			summary_writer.add_summary(summary,step)
			print("After %d steps training ,the loss is %g"%(step, loss))

			if step % SAVE_STEP == 0 and SAVE:
				saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME), global_step = global_step)
				print(" ")
				print('==================================================')
				print("After %d steps training, model save succeed" %(step))
				print('==================================================')
				print(" ")
		
		summary_writer.close()



# -------------------------------------------------



def Calculate(str):
	length = len(str)
	str1 = ''
	for i in range(length):
		if ord(str[i]) < 48:
			str1 += str[i]
	po1 = str1[0]
	po2 = str1[-1]
	a = int(str.split(po1)[0])
	tmp = str[str.find(po1)+1:length]
	b =int(tmp.split(po2)[0])
	c = int(tmp.split(po2)[-1])
	if po1 == '+' and po2 == '+':
		return a + b + c
	if po1 == '+' and po2 == '-':
		return a + b - c
	if po1 == '+' and po2 == '*':
		return a + b * c
	if po1 == '-' and po2 == '+':
		return a - b + c
	if po1 == '-' and po2 == '-':
		return a - b - c	
	if po1 == '-' and po2 == '*':
		return a - b * c	
	if po1 == '*' and po2 == '+':
		return a * b + c	
	if po1 == '*' and po2 == '-':
		return a * b - c	
	if po1 == '*' and po2 == '*':
		return a * b * c	

def create_test_images_lists():

    result = {}

    extension = 'jpg'
    dir_name = 'test'
    file_list = []
    file_glob = os.path.join(INPUT_DATA, dir_name, "*." + extension)
    file_list.extend(glob.glob(file_glob))

    images = []
    for file_name in file_list:
        base_name = os.path.basename(file_name)
        images.append(base_name)
    result[dir_name] = images
    
    return result


def output2char(num):
	if num == 0:
		return '0'
	elif num == 1:
		return '1'
	elif num == 2:
		return '2'
	elif num == 3:
		return '3'
	elif num == 4:
		return '4'
	elif num == 5:
		return '5'
	elif num == 6:
		return '6'
	elif num == 7:
		return '7'
	elif num == 8:
		return '8'
	elif num == 9:
		return '9'
	elif num == 10:
		return '+'
	elif num == 11:
		return '-'	
	elif num == 12:
		return '*'	


def test_crack_captcha_cnn():

	file = open(RESULT_DIR, 'w')
	output = crack_captcha_cnn()
	predict = tf.argmax(output, 1)
	saver = tf.train.Saver()
	correct_num = 0
	error = []
	test_image_dict = create_test_images_lists()

	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess,ckpt.model_checkpoint_path)
			print("Load successfully!")
		else:
			print("No found checkpoint!")

		for i in range(SIZE_TEST_SET):


			img_name = test_image_dict['test'][i]
			image_file = os.path.join(INPUT_DATA_TEST, img_name)
			img = Image.open(image_file)
			img = image_process.FinalProcess(img)
			images = image_process.crop_for_test(img, image_process.segmentation(img))
			
			correct = label_dict[img_name.split('.')[0]]
			char_list = []
			size = len(images)

			batch_test = np.zeros([size, IMAGE_HEIGHT * IMAGE_WIDTH])
			index = 0

			for image in images:
				image = np.array(image)
				image = image.flatten() / 255
				batch_test[index, :] = image
				index += 1

			char_pos = sess.run(predict, feed_dict = {X: batch_test, keep_prob: 1.0})
			for char_ in char_pos:
				char_list.append(output2char(char_))
			
			result = "False"
			prediction = "".join(char_list) + '=' + str(Calculate("".join(char_list)))
			if (correct == prediction):
				correct_num += 1
				result = "True"
			else: 
				result = "False"
				error.append(i)
			print("No.%s  正确: %-14s  预测: %-14s  结果： %s" %(str(i), correct, prediction, result))
			file.write("%0.4d" % i + "," + "".join(char_list) + '=' + str(Calculate("".join(char_list))) + '\n')

		print(" ")
		print('======================================')
		print("The final accurancy in the test set is %g%%."%(correct_num * 100. / SIZE_TEST_SET))		
		print('======================================')
		print(" ")

		print("The index of error is ",end = "")
		print(error)

	file.close()

if __name__ == '__main__':
	if MODE == "train":
		train_crack_captcha_cnn()
	elif MODE == "test":
		test_crack_captcha_cnn()
