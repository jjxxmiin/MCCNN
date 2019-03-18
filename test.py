import tensorflow as tf
import utils
import ops
from tensorflow.contrib import slim
import mcnn
import cv2
from utils import load,show_density_map

file = "IMG_23"

checkpoint_dir = "checkpoint"
test_img_path ="G:/ShanghaiTech/part_B/test_data/images/" + file + ".jpg"
test_dmp_path ="G:/ShanghaiTech/part_B/test_data/ground-truth/GT_" + file + ".mat"

img, gt_dmp, gt_count = utils.read_test_data(test_img_path, test_dmp_path, scale=4)

test = tf.placeholder(tf.float32,shape=[None,None,None,3])

estimate = mcnn.multi_column_cnn(test)
saver = tf.train.Saver()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    could_load, checkpoint_counter = load(checkpoint_dir, sess, saver)

    if could_load:
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")
        exit(0)

    img = utils.input_normalization(img)

    prediction= sess.run(estimate, feed_dict={test: img})

    print(prediction.sum())
    print(gt_count)

    #show_density_map(gt_dmp[-1,:,:,-1])



