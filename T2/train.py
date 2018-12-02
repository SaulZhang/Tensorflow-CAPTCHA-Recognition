import numpy as np
import tensorflow as tf
import os 
import glob
import string
from PIL import Image
import random
import tensorflow.contrib.slim as slim
import image_process


MODE = "test"
RESTORE = True
SAVE = True
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
SIZE_TRAIN_SET = 9900
SIZE_TEST_SET = 5000
TRAINING_STEPS = 20000 + 5


MODEL_SAVE_PATH = "./model"
MODEL_NAME = "model.ckpt"
SUMMARY_DIR ="./log"
INPUT_DATA_TRAIN = "./process_data/train/"
INPUT_DATA_TEST = "C:/Users/Jet Zhang/Desktop/测试数据第一类第二类/test-2/test"
INPUT_DATA = "C:/Users/Jet Zhang/Desktop/测试数据第一类第二类/test-2/"
LABEL_DIR = "./process_data/mappings.txt"
RESULT_DIR = './result/mappings.txt'


IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
BATCH_SIZE = 128
OUTPUT_SIZE = 36
SAVE_STEP = 500


FOOT_0 = 0
Total_FOOT_0 = 0
FOOT_1 = 0
Total_FOOT_1 = 0 
FOOT_2 = 0
Total_FOOT_2 = 920
FOOT_3 = 0
Total_FOOT_3 = 959
FOOT_4 = 0
Total_FOOT_4 = 836
FOOT_5 = 0
Total_FOOT_5 = 890
FOOT_6 = 0
Total_FOOT_6 = 902
FOOT_7 = 0
Total_FOOT_7 = 894
FOOT_8 = 0
Total_FOOT_8 = 944
FOOT_9 = 0
Total_FOOT_9 = 884
FOOT_A = 0
Total_FOOT_A = 1858
FOOT_B = 0
Total_FOOT_B = 1851
FOOT_C = 0
Total_FOOT_C = 1840
FOOT_D = 0
Total_FOOT_D = 1901
FOOT_E = 0
Total_FOOT_E = 1849
FOOT_F = 0
Total_FOOT_F = 1875
FOOT_G = 0
Total_FOOT_G = 1810
FOOT_H = 0
Total_FOOT_H = 1787
FOOT_I = 0
Total_FOOT_I = 0
FOOT_J = 0
Total_FOOT_J = 1841
FOOT_K = 0
Total_FOOT_K = 1883
FOOT_L = 0
Total_FOOT_L = 954
FOOT_M = 0
Total_FOOT_M = 1790
FOOT_N = 0
Total_FOOT_N = 1917
FOOT_O = 0
Total_FOOT_O = 0
FOOT_P = 0
Total_FOOT_P = 1840
FOOT_Q = 0
Total_FOOT_Q = 1799
FOOT_R = 0
Total_FOOT_R = 1806
FOOT_S = 0
Total_FOOT_S = 1821
FOOT_T = 0
Total_FOOT_T = 918
FOOT_U = 0
Total_FOOT_U = 1853
FOOT_V = 0
Total_FOOT_V = 1911
FOOT_W = 0
Total_FOOT_W = 1791
FOOT_X = 0
Total_FOOT_X = 1787
FOOT_Y = 0
Total_FOOT_Y = 1809
FOOT_Z = 0
Total_FOOT_Z = 1780

TOTAL_NUM_EXAMPLE = 49500

char_set = string.digits + string.ascii_uppercase
CHAR_SET_LEN = len(char_set) 


def text2vec(char):
	pos = 0 
	vector = np.zeros(CHAR_SET_LEN)
	if  48 <= ord(char) and ord(char) <= 57:
		pos = ord(char) - 48
	if  65 <= ord(char) and ord(char) <= 90:
		pos = ord(char) - 55
	vector[pos] = 1
	return vector


def vec2text(vec):
	char_pos = vec.nonzero()[0]
	if 0 <= char_pos and char_pos <= 9:
		char_code = char_pos + ord('0')
	if 10 <= char_pos and char_pos <= 35:
		char_pos -= 10
		char_code = char_pos + ord('A')
	return char_code


def get_next_batch_train(batch_size=128):
	batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
	batch_y = np.zeros([batch_size, CHAR_SET_LEN])

	global FOOT_0, Total_FOOT_0, FOOT_1, Total_FOOT_1, FOOT_2, Total_FOOT_2, FOOT_3, Total_FOOT_3, FOOT_4, Total_FOOT_4
	global FOOT_5, Total_FOOT_5, FOOT_6, Total_FOOT_6, FOOT_7, Total_FOOT_7, FOOT_8, Total_FOOT_8, FOOT_9, Total_FOOT_9
	global FOOT_A, Total_FOOT_A, FOOT_B, Total_FOOT_B, FOOT_C, Total_FOOT_C, FOOT_D, Total_FOOT_D, FOOT_E, Total_FOOT_E
	global FOOT_F, Total_FOOT_F, FOOT_G, Total_FOOT_G, FOOT_H, Total_FOOT_H, FOOT_I, Total_FOOT_I, FOOT_J, Total_FOOT_J
	global FOOT_K, Total_FOOT_K, FOOT_L, Total_FOOT_L, FOOT_M, Total_FOOT_M, FOOT_N, Total_FOOT_N, FOOT_O, Total_FOOT_O
	global FOOT_P, Total_FOOT_P, FOOT_Q, Total_FOOT_Q, FOOT_R, Total_FOOT_R, FOOT_S, Total_FOOT_S, FOOT_T, Total_FOOT_T
	global FOOT_U, Total_FOOT_U, FOOT_V, Total_FOOT_V, FOOT_W, Total_FOOT_W, FOOT_X, Total_FOOT_X, FOOT_Y, Total_FOOT_Y
	global FOOT_Z , Total_FOOT_Z

	Totalcount = 1  
	count = Totalcount - 1 
	while True:
		count = Totalcount - 1
		rand = random.randint(1,36)

		while rand == 1 or rand == 2 or rand == 19 or rand == 25:
			rand = random.randint(1,36)
		if rand == 3:
			if FOOT_2 == Total_FOOT_2:
				FOOT_2 %= Total_FOOT_2
			path = os.path.join(INPUT_DATA_TRAIN, '2', train_image_dict['2'][FOOT_2])
			FOOT_2 += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1 
			batch_y[count, :] = text2vec('2')
		elif rand == 4:
			if FOOT_3 == Total_FOOT_3:
				FOOT_3 %= Total_FOOT_3
			path = os.path.join(INPUT_DATA_TRAIN, '3', train_image_dict['3'][FOOT_3])
			FOOT_3 += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1
			batch_y[count, :] = text2vec('3')
		elif rand == 5:
			if FOOT_4 == Total_FOOT_4:
				FOOT_4 %= Total_FOOT_4
			path = os.path.join(INPUT_DATA_TRAIN, '4', train_image_dict['4'][FOOT_4])
			FOOT_4 += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1
			batch_y[count, :] = text2vec('4')
		elif rand == 6:
			if FOOT_5 == Total_FOOT_5:
				FOOT_5 %= Total_FOOT_5
			path = os.path.join(INPUT_DATA_TRAIN, '5', train_image_dict['5'][FOOT_5])
			FOOT_5 += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1
			batch_y[count, :] = text2vec('5')
		elif rand == 7:
			if FOOT_6 == Total_FOOT_6:
				FOOT_6 %= Total_FOOT_6
			path = os.path.join(INPUT_DATA_TRAIN, '6', train_image_dict['6'][FOOT_6])
			FOOT_6 += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1
			batch_y[count, :] = text2vec('6')
		elif rand == 8:
			if FOOT_7 == Total_FOOT_7:
				FOOT_7 %= Total_FOOT_7
			path = os.path.join(INPUT_DATA_TRAIN, '7', train_image_dict['7'][FOOT_7])
			FOOT_7 += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1 
			batch_y[count, :] = text2vec('7')
		elif rand == 9:
			if FOOT_8 == Total_FOOT_8:
				FOOT_8 %= Total_FOOT_8
			path = os.path.join(INPUT_DATA_TRAIN, '8', train_image_dict['8'][FOOT_8])
			FOOT_8 += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1
			batch_y[count, :] = text2vec('8')
		elif rand == 10:
			if FOOT_9 == Total_FOOT_9:
				FOOT_9 %= Total_FOOT_9
			path = os.path.join(INPUT_DATA_TRAIN, '9', train_image_dict['9'][FOOT_9])
			FOOT_9 += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1
			batch_y[count, :] = text2vec('9')
		elif rand == 11:
			if FOOT_A == Total_FOOT_A:
				FOOT_A %= Total_FOOT_A
			path = os.path.join(INPUT_DATA_TRAIN, 'A', train_image_dict['A'][FOOT_A])
			FOOT_A += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1
			batch_y[count, :] = text2vec('A')
		elif rand == 12:
			if FOOT_B == Total_FOOT_B:
				FOOT_B %= Total_FOOT_B
			path = os.path.join(INPUT_DATA_TRAIN, 'B', train_image_dict['B'][FOOT_B])
			FOOT_B += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1
			batch_y[count, :] = text2vec('B')
		elif rand == 13:
			if FOOT_C == Total_FOOT_C:
				FOOT_C %= Total_FOOT_C
			path = os.path.join(INPUT_DATA_TRAIN, 'C', train_image_dict['C'][FOOT_C])
			FOOT_C += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1
			batch_y[count, :] = text2vec('C')
		elif rand == 14:
			if FOOT_D == Total_FOOT_D:
				FOOT_D %= Total_FOOT_D
			path = os.path.join(INPUT_DATA_TRAIN, 'D', train_image_dict['D'][FOOT_D])
			FOOT_D += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1 
			batch_y[count, :] = text2vec('D')
		elif rand == 15:
			if FOOT_E == Total_FOOT_E:
				FOOT_E %= Total_FOOT_E
			path = os.path.join(INPUT_DATA_TRAIN, 'E', train_image_dict['E'][FOOT_E])
			FOOT_E += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count,: ] = image.flatten() / 128 - 1
			batch_y[count,: ] = text2vec('E')
		elif rand == 16:
			if FOOT_F == Total_FOOT_F:
				FOOT_F %= Total_FOOT_F
			path = os.path.join(INPUT_DATA_TRAIN, 'F', train_image_dict['F'][FOOT_F])
			FOOT_F += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1
			batch_y[count, :] = text2vec('F')
		elif rand == 17:
			if FOOT_G == Total_FOOT_G:
				FOOT_G %= Total_FOOT_G
			path = os.path.join(INPUT_DATA_TRAIN, 'G', train_image_dict['G'][FOOT_G])
			FOOT_G += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1
			batch_y[count, :] = text2vec('G')
		elif rand == 18:
			if FOOT_H == Total_FOOT_H:
				FOOT_H %= Total_FOOT_H
			path = os.path.join(INPUT_DATA_TRAIN, 'H', train_image_dict['H'][FOOT_H])
			FOOT_H += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1
			batch_y[count, :] = text2vec('H')
	
		elif rand == 20:
			if FOOT_J == Total_FOOT_J:
				FOOT_J %= Total_FOOT_J
			path = os.path.join(INPUT_DATA_TRAIN, 'J', train_image_dict['J'][FOOT_J])
			FOOT_J += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1
			batch_y[count, :] = text2vec('J')
		elif rand == 21:
			if FOOT_K == Total_FOOT_K:
				FOOT_K %= Total_FOOT_K
			path = os.path.join(INPUT_DATA_TRAIN, 'K', train_image_dict['K'][FOOT_K])
			FOOT_K += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1
			batch_y[count, :] = text2vec('K')
		elif rand == 22:
			if FOOT_L == Total_FOOT_L:
				FOOT_L %= Total_FOOT_L
			path = os.path.join(INPUT_DATA_TRAIN, 'L', train_image_dict['L'][FOOT_L])
			FOOT_L += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1
			batch_y[count, :] = text2vec('L')
		elif rand == 23:
			if FOOT_M == Total_FOOT_M:
				FOOT_M %= Total_FOOT_M
			path = os.path.join(INPUT_DATA_TRAIN, 'M', train_image_dict['M'][FOOT_M])
			FOOT_M += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1
			batch_y[count, :] = text2vec('M')
		elif rand == 24:
			if FOOT_N == Total_FOOT_N:
				FOOT_N %= Total_FOOT_N
			path = os.path.join(INPUT_DATA_TRAIN, 'N', train_image_dict['N'][FOOT_N])
			FOOT_N += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1
			batch_y[count, :] = text2vec('N')
 
		elif rand == 26:
			if FOOT_P == Total_FOOT_P:
				FOOT_P %= Total_FOOT_P
			path = os.path.join(INPUT_DATA_TRAIN, 'P', train_image_dict['P'][FOOT_P])
			FOOT_P += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1
			batch_y[count, :] = text2vec('P')
		elif rand == 27:
			if FOOT_Q == Total_FOOT_Q:
				FOOT_Q %= Total_FOOT_Q
			path = os.path.join(INPUT_DATA_TRAIN, 'Q', train_image_dict['Q'][FOOT_Q])
			FOOT_Q += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1
			batch_y[count, :] = text2vec('Q')
		elif rand == 28:
			if FOOT_R == Total_FOOT_R:
				FOOT_R %= Total_FOOT_R
			path = os.path.join(INPUT_DATA_TRAIN, 'R', train_image_dict['R'][FOOT_R])
			FOOT_R += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1
			batch_y[count, :] = text2vec('R')
		elif rand == 29:
			if FOOT_S == Total_FOOT_S:
				FOOT_S %= Total_FOOT_S
			path = os.path.join(INPUT_DATA_TRAIN, 'S', train_image_dict['S'][FOOT_S])
			FOOT_S += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1
			batch_y[count, :] = text2vec('S')
		elif rand == 30:
			if FOOT_T == Total_FOOT_T:
				FOOT_T %= Total_FOOT_T
			path = os.path.join(INPUT_DATA_TRAIN, 'T', train_image_dict['T'][FOOT_T])
			FOOT_T += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1
			batch_y[count, :] = text2vec('T')
		elif rand == 31:
			if FOOT_U == Total_FOOT_U:
				FOOT_U %= Total_FOOT_U
			path = os.path.join(INPUT_DATA_TRAIN, 'U', train_image_dict['U'][FOOT_U])
			FOOT_U += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1 
			batch_y[count, :] = text2vec('U')
		elif rand == 32:
			if FOOT_V == Total_FOOT_V:
				FOOT_V %= Total_FOOT_V
			path = os.path.join(INPUT_DATA_TRAIN, 'V', train_image_dict['V'][FOOT_V])
			FOOT_V += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1
			batch_y[count, :] = text2vec('V')
		elif rand == 33:
			if FOOT_W == Total_FOOT_W:
				FOOT_W %= Total_FOOT_W
			path = os.path.join(INPUT_DATA_TRAIN, 'W', train_image_dict['W'][FOOT_W])
			FOOT_W += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1
			batch_y[count, :] = text2vec('W')
		elif rand == 34:
			if FOOT_X == Total_FOOT_X:
				FOOT_X %= Total_FOOT_X
			path = os.path.join(INPUT_DATA_TRAIN, 'X', train_image_dict['X'][FOOT_X])
			FOOT_X += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1
			batch_y[count, :] = text2vec('X')
		elif rand == 35:
			if FOOT_Y == Total_FOOT_Y:
				FOOT_Y %= Total_FOOT_Y
			path = os.path.join(INPUT_DATA_TRAIN, 'Y', train_image_dict['Y'][FOOT_Y])
			FOOT_Y += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1
			batch_y[count, :] = text2vec('Y')
		elif rand == 36:
			if FOOT_Z == Total_FOOT_Z:
				FOOT_Z %= Total_FOOT_Z
			path = os.path.join(INPUT_DATA_TRAIN, 'Z', train_image_dict['Z'][FOOT_Z])
			FOOT_Z += 1
			image = Image.open(path)
			image = np.array(image)
			batch_x[count, :] = image.flatten() / 128 - 1
			batch_y[count, :] = text2vec('Z')

		Totalcount += 1
		if Totalcount == batch_size + 1:
			break
	return batch_x, batch_y
 

def crack_captcha_cnn():
	x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

	conv_1 = slim.conv2d(x, 64, [3, 3], 1, padding ='SAME', scope = 'conv1', activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer_conv2d(), reuse=None)
	max_pool_1 = slim.max_pool2d(conv_1, [2, 2], [2, 2], padding = 'SAME')
	conv_2 = slim.conv2d(max_pool_1, 128, [3, 3], padding = 'SAME', scope ='conv2', activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer_conv2d())
	max_pool_2 = slim.max_pool2d(conv_2, [2, 2], [2, 2], padding = 'SAME')
	conv_3 = slim.conv2d(max_pool_2, 256, [3, 3], padding = 'SAME', scope ='conv3', activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer_conv2d())
	max_pool_3 = slim.max_pool2d(conv_3, [2, 2], [2, 2], padding = 'SAME')

	flatten = slim.flatten(max_pool_3)
	fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024, activation_fn = tf.nn.relu, scope = 'fc1')
	logits = slim.fully_connected(slim.dropout(fc1, keep_prob), OUTPUT_SIZE, activation_fn = None, scope = 'fc2')

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
        value = value[0: 5]
        label_dict[key] = value
    file.close()

    return label_dict


train_image_dict = create_train_images_lists()
label_dict = create_labels_lists()

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)


def train_crack_captcha_cnn():
	with tf.device("/cpu:0"):
		output = crack_captcha_cnn()
		global_step = tf.Variable(0, trainable = False)

		with tf.name_scope('loss'):
			loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = output, labels = Y))
			tf.summary.scalar('loss', loss)
		
		learning_rate = tf.train.exponential_decay(
			LEARNING_RATE_BASE,
			global_step,
			TOTAL_NUM_EXAMPLE / BATCH_SIZE,
			LEARNING_RATE_DECAY
			)

		optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = global_step)

		merged = tf.summary.merge_all()
		saver = tf.train.Saver()
		with tf.Session() as sess:
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
				summary, _ , loss_ , step = sess.run([merged, optimizer, loss, global_step], feed_dict = {X: batch_x, Y: batch_y, keep_prob: 0.8})
				summary_writer.add_summary(summary,step)
				print("After %d steps training ,the loss is %g" %(step,loss_))
				
				if step % SAVE_STEP == 0 and SAVE:
					saver.save(sess,os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step)
					print(" ")
					print('==================================================')
					print("After %d steps training, model save succeed" %(step))
					print('==================================================')
					print(" ")

			summary_writer.close()



# -------------------------------------------------



def output2char(num):
	if   num == 0:
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
		return 'A'
	elif num == 11:
		return 'B'	
	elif num == 12:
		return 'C'	
	elif num == 13:
		return 'D'
	elif num == 14:
		return 'E'
	elif num == 15:
		return 'F'
	elif num == 16:
		return 'G'
	elif num == 17:
		return 'H'
	elif num == 18:
		return 'I'
	elif num == 19:
		return 'J'
	elif num == 20:
		return 'K'
	elif num == 21:
		return 'L'
	elif num == 22:
		return 'M'
	elif num == 23:
		return 'N'	
	elif num == 24:
		return 'O'	
	elif num == 25:
		return 'P'
	elif num == 26:
		return 'Q'
	elif num == 27:
		return 'R'
	elif num == 28:
		return 'S'
	elif num == 29:
		return 'T'
	elif num == 30:
		return 'U'
	elif num == 31:
		return 'V'
	elif num == 32:
		return 'W'
	elif num == 33:
		return 'X'
	elif num == 34:
		return 'Y'
	elif num == 35:
		return 'Z'	

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



def test_crack_captcha_cnn():
	file = open(RESULT_DIR,'w')
	output = crack_captcha_cnn()
	predict = tf.argmax(output, 1)
	correct_num = 0
	# error = []
	saver = tf.train.Saver()
	test_image_dict = create_test_images_lists()

	# print(test_image_dict)
	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print("Load successfully!")
		else:
			print("No found checkpoint!")

		for i in range(SIZE_TEST_SET):
			print(i)
			img_name = test_image_dict['test'][i]
			image_file = os.path.join(INPUT_DATA_TEST, img_name)
			img = Image.open(image_file)
			img = image_process.FinalProcess(img)
			images = image_process.crop_for_test(img, image_process.segmentation(img))
			
			# correct = label_dict[img_name.split('.')[0]]
			char_list = []
			size = len(images)

			batch_test = np.zeros([size, IMAGE_HEIGHT * IMAGE_WIDTH])
			index = 0

			for image in images:
				image = np.array(image)
				image = image.flatten() / 128 - 1
				batch_test[index, :] = image
				index += 1

			char_pos = sess.run(predict, feed_dict = {X: batch_test, keep_prob: 1.0})
			for char_ in char_pos:
				char_list.append(output2char(char_))

			prediction = "".join(char_list)
			# result = "False"
			# if (correct == prediction):
			# 	result = "True"
			# 	correct_num += 1
			# else: 
			# 	result = "False"
			#error.append(i)
			file.write("%0.4d" % i + "," + "".join(char_list) + '\n')
		# 	print("No.{}  正确: {}  预测: {}  结果： {}".format(str(i), correct, prediction, result))

		# print(" ")
		# print('======================================')
		# print("The final accurancy in the test set is %g%%." %(correct_num * 100. / SIZE_TEST_SET))	
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
