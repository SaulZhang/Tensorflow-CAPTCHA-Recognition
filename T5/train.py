import tensorflow as tf
import numpy as np
import random
import os
import glob
from PIL import Image
import tensorflow.contrib.slim as slim
import image_process


MODE = 'test'
RESTORE = True
SAVE = True
SIZE_TEST_SET = 100
SIZE_TRAIN_SET = 9900
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
TRAINING_STEPS = 500000 + 5


MODEL_SAVE_PATH = "./model"
MODEL_NAME = "model.ckpt"
SUMMARY_DIR = "./log"
RESULT_DIR = "./result/mappings.txt"
INPUT_DATA_TRAIN = "./process_data/train/"
INPUT_DATA_TEST = "./process_data/test/"
LABEL_DIR = "./process_data/mappings.txt"


BTACH_SIZE = 128
IMAGE_HEIGHT = 45
IMAGE_WIDTH = 45
MIN = 0
OUTPUT_SIZE = 128
Per_Pic = 32
SAVE_STEP = 500
TEST_STEP = 1000


FOOT_TRAIN = 0
FOOT_TEST = 0

delta = 30

def enhance_data(img):
    img = np.array(img)
    img = -1 * img + 255
    img = Image.fromarray(img)
    img = img.rotate(random.randint(-delta, delta), expand = 0)
    img = np.array(img)
    img = -1 * img + 255
    img = Image.fromarray(img)
    return img


class SIAMESE(object):

    def siamese(self, inputs, reuse = False):
        
        conv_1 = slim.conv2d(inputs, 64, [3, 3], 1, padding = 'SAME', scope = 'conv1', activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer_conv2d(), reuse = reuse)
        max_pool_1 = slim.max_pool2d(conv_1, [2, 2], [2, 2], padding = 'SAME')
        conv_2 = slim.conv2d(max_pool_1, 128, [3, 3], padding = 'SAME', scope = 'conv2', activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer_conv2d(), reuse = reuse)
        max_pool_2 = slim.max_pool2d(conv_2, [2, 2], [2, 2], padding = 'SAME')
        conv_3 = slim.conv2d(max_pool_2, 256, [3, 3], padding = 'SAME', scope = 'conv3', activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer_conv2d(), reuse = reuse)
        max_pool_3 = slim.max_pool2d(conv_3, [2, 2], [2, 2], padding = 'SAME')

        flatten = slim.flatten(max_pool_3)
        fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024, activation_fn = tf.nn.relu, scope = 'fc1')
        logits = slim.fully_connected(slim.dropout(fc1, keep_prob), OUTPUT_SIZE, activation_fn = None, scope = 'fc2')

        return logits


    def siamese_loss(self, out1, out2, y):
        
        output_difference = tf.abs(out1 - out2)

        W = tf.Variable(tf.random_normal([OUTPUT_SIZE, 1], stddev = 0.1),name = 'W')
        b = tf.Variable(tf.zeros(1, 1) + 0.1, name = 'b')
        y_ = tf.nn.sigmoid(tf.matmul(output_difference, W) + b, name = "distance") 
        losses = -((y * tf.log(y_ + 1e-9)) + (1 - y) * tf.log(1 - y_ + 1e-9))
        loss = tf.reduce_mean(losses)
        
        return out1 , out2 , y_ , loss

def get_next_batch_train(batch_size = BTACH_SIZE):
    batch_x1 = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH]) 
    batch_x2 = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH]) 
    batch_y = np.zeros([batch_size, 1])
    global FOOT_TRAIN

    count = 0

    while True:
    
        if FOOT_TRAIN % SIZE_TRAIN_SET == 0:
            FOOT_TRAIN = 0 
        
        num = "%0.4d" %FOOT_TRAIN    
        relation_pos = []
        
        for i in range(4):
            pos = int(label_dict[num][i])
            relation_pos.append(pos)

        for i in range(4):
            for j in range(8):
                img_name = train_image_dict[num][i + 9]
                image_file = os.path.join(INPUT_DATA_TRAIN, num, img_name)
                image = Image.open(image_file)
                image = enhance_data(image)
                image = np.array(image)
                image = (image.flatten() / 128.) - 1.
                batch_x1[count, :] = image

                img_name = train_image_dict[num][relation_pos[i]]
                image_file = os.path.join(INPUT_DATA_TRAIN, num, img_name)
                image = Image.open(image_file)
                image = enhance_data(image)
                image = np.array(image)
                image = (image.flatten() / 128.) - 1.
                batch_x2[count, :] = image

                batch_y[count] = np.array([1.0])
                count += 1

        for i in range(4):
            for j in range(9):
                if j == relation_pos[i]:
                    continue

                img_name = train_image_dict[num][i + 9]
                image_file = os.path.join(INPUT_DATA_TRAIN, num, img_name)
                image = Image.open(image_file)
                image = enhance_data(image)
                image = np.array(image)
                image = (image.flatten() / 128.) - 1.
                batch_x1[count, :] = image

                img_name = train_image_dict[num][j]
                image_file = os.path.join(INPUT_DATA_TRAIN, num, img_name)
                image = Image.open(image_file)
                image = enhance_data(image)
                image = np.array(image)
                image = (image.flatten() / 128.) - 1.
                batch_x2[count, :] = image

                batch_y[count] = np.array([0.0])
                count += 1

        FOOT_TRAIN += 1
        if count == batch_size:
            break

    cc = list(zip(batch_x1, batch_x2, batch_y))
    random.shuffle(cc)
    batch_x11, batch_x22, batch_y33 = zip(*cc)
    return batch_x11, batch_x22, batch_y33

def create_train_images_lists():

    result = {}
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA_TRAIN)]
    is_root_dir = True
    count = 0 
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
    file = open(LABEL_DIR,'r')
    label_dict = {}
    for line in file:
        key, value = line.split(",")
        value = value[0: len(value) - 1]
        label_dict[key] = value
    file.close()

    return label_dict


train_image_dict = create_train_images_lists()
label_dict = create_labels_lists()

with tf.variable_scope('global_step') as scope:
    global_step = tf.Variable(0,trainable=False)

with tf.name_scope("input"):
    with tf.variable_scope('input_x1') as scope:
        x1 = tf.placeholder(tf.float32, shape = [None, IMAGE_HEIGHT * IMAGE_WIDTH])
    with tf.name_scope('x1-reshaped'):
        x_input_1 = tf.reshape(x1, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    with tf.variable_scope('input_x2') as scope:
        x2 = tf.placeholder(tf.float32, shape = [None, IMAGE_HEIGHT * IMAGE_WIDTH])
    with tf.name_scope('x2-reshaped'):
        x_input_2 = tf.reshape(x2, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    with tf.variable_scope('y') as scope:
        y = tf.placeholder(tf.float32, shape = [None, 1])

    with tf.variable_scope('keep_prob') as scope:
        keep_prob = tf.placeholder(tf.float32)


with tf.variable_scope('siamese') as scope:
    out1 = SIAMESE().siamese(x_input_1, reuse = False)
    scope.reuse_variables()
    out2 = SIAMESE().siamese(x_input_2, reuse = True)

with tf.variable_scope('Learning_Rate') as scope:
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        Per_Pic * SIZE_TRAIN_SET / BTACH_SIZE,
        LEARNING_RATE_DECAY
        )
    tf.summary.scalar('Learning_Rate',learning_rate)

with tf.variable_scope('Contractive_loss') as scope:
    model1,model2,distance,Contractive_loss = SIAMESE().siamese_loss(out1, out2, y)
    
with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(Contractive_loss, global_step = global_step)

loss_summary = tf.summary.scalar('loss', Contractive_loss)

def train_crack_captcha_cnn():

    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
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
            X1_, X2_, Y_ = get_next_batch_train()
            _, loss_, summ, step = sess.run([train_op, Contractive_loss, merged, global_step],
                                    feed_dict={x1: X1_, x2: X2_, y: Y_, keep_prob: 0.8})

            summary_writer.add_summary(summ,step)
            
            print("After %d steps training, loss is %g"%(step, loss_))

            if step % SAVE_STEP == 0 and SAVE: 
                saver.save(sess,os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step)
                print(" ")
                print('==================================================')
                print("After %d steps training, model save succeed" %(step))
                print('==================================================')
                print(" ")


        summary_writer.close()

def create_test_dir_lists():
#把子文件夹的名字存进列表里面

    result = []
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA_TEST)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        dir_name = os.path.basename(sub_dir)
        result.append(dir_name)
    return result


def test_crack_captcha_cnn():
    test_dir = create_test_dir_lists()
    # error= []
    saver = tf.train.Saver()
    file = open(RESULT_DIR, 'w')
    correct_num = 0

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Load successfully!")
        else:
            print("No checkpoint file found")

        extension = '.jpg'

        for num in range(SIZE_TEST_SET):
            sub_dir = test_dir[num]
            batch_validation = np.zeros([13, IMAGE_HEIGHT * IMAGE_WIDTH])

            for i in range(9):
                img_name = str(i) + extension
                path = os.path.join(INPUT_DATA_TEST, sub_dir, img_name)
                image = Image.open(path)
                image = image_process.FinalProcess(image)
                image = np.array(image)
                image = (image.flatten() / 128.) - 1.
                batch_validation[i, :] = image

            img_name = sub_dir + extension
            path = os.path.join(INPUT_DATA_TEST, sub_dir, img_name)
            image = Image.open(path)
            image = image_process.FinalProcess(image)

            img_crop = image.crop((0, 0, 45, 45))
            img_crop = image_process.FinalProcess(img_crop)
            img_crop = np.array(img_crop)
            img_crop = (img_crop.flatten() / 128.) - 1.
            batch_validation[9, :] = img_crop

            img_crop = image.crop((34, 0, 79, 45))
            img_crop = image_process.FinalProcess(img_crop)
            img_crop = np.array(img_crop)
            img_crop = (img_crop.flatten() / 128.) - 1.
            batch_validation[10, :] = img_crop
            
            img_crop = image.crop((71, 0, 116, 45))
            img_crop = image_process.FinalProcess(img_crop)
            img_crop = np.array(img_crop)
            img_crop = (img_crop.flatten() / 128.) - 1.
            batch_validation[11, :] = img_crop
            
            img_crop = image.crop((105, 0, 150, 45))
            img_crop = image_process.FinalProcess(img_crop)
            img_crop = np.array(img_crop)
            img_crop = (img_crop.flatten() / 128.) - 1.
            batch_validation[12, :] = img_crop


            batch_x1 = np.zeros([36, IMAGE_HEIGHT * IMAGE_WIDTH])
            batch_x2 = np.zeros([36, IMAGE_HEIGHT * IMAGE_WIDTH])
            index = 0;
            for i in range(4):
                for j in range(9):
                    batch_x1[index, :] = batch_validation[i + 9, :]
                    batch_x2[index, :] = batch_validation[j, :]
                    index += 1;
            
            dist = distance.eval({x_input_1: batch_x1.reshape([-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1]),x_input_2: batch_x2.reshape([-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1]), keep_prob: 1.0})
            dist = dist.reshape((4, 9))
            cur_option = np.argmax(dist, 1)

            prediction = str(cur_option[0]) + str(cur_option[1]) + str(cur_option[2]) + str(cur_option[3])
            
            # result = "False"
            # correct = label_dict[sub_dir][0: 4]
            # if prediction == correct:
            #     correct_num += 1
            #     result = "True"
            # else:
            #     result = "False"
            #     error.append(num)

            # print("No.{}  正确: {}  预测: {}  结果： {}".format(num, correct, prediction, result))
            file.write("%0.4d" % num+ "," + "".join(prediction) + '\n')            
            
        # print(" ")
        # print('======================================')
        # print("The final accurancy in the test set is %g%%"%(correct_num * 100. / SIZE_TEST_SET))
        # print('======================================')
        # print(" ")

        # print("The index of error is ", end = "")
        # print(error)

    file.close()

if __name__ == '__main__':
    if MODE == 'train':
        train_crack_captcha_cnn()
    elif MODE == 'test':
        test_crack_captcha_cnn()