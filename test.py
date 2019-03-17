import tensorflow as tf
import mcnn
import cv2
from utils import load,show_density_map

checkpoint_dir = "checkpoint"
test_img_path ="G:/ShanghaiTech/part_B/test_data/images/IMG_20.jpg"

test_img = cv2.imread(test_img_path)

test = tf.placeholder(tf.float32,[1,None,None,3])

with tf.Session() as sess:
    saver = tf.train.Saver()

    could_load, checkpoint_counter = load(checkpoint_dir, sess, saver)

    if could_load:
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")
        exit(0)

    estimate = mcnn.multi_column_cnn(test,scope='test')

    prediction= sess.run([estimate], feed_dict={test: test_img})

    print(prediction.sum())
    
    show_density_map(prediction)