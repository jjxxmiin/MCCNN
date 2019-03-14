from ops import *
import tensorflow as tf
from tensorflow.contrib import slim


def multi_column_cnn(inputs, scope='mcnn'):
    with tf.variable_scope(scope):
        with tf.variable_scope('large'):
            large_column = slim.conv2d(inputs, 16, [9, 9], padding='SAME', scope='conv_1_9x9')
            large_column = relu(large_column)
            large_column = slim.conv2d(large_column, 32, [7, 7], padding='SAME', scope='conv2_9x9')
            large_column = relu(large_column)
            large_column = slim.max_pool2d(large_column, [2, 2], 2, scope='pool1_9x9')
            large_column = slim.conv2d(large_column, 16, [7, 7], padding='SAME', scope='conv3_9x9')
            large_column = relu(large_column)
            large_column = slim.max_pool2d(large_column, [2, 2], 2, scope='pool2_9x9')
            large_column = slim.conv2d(large_column, 8, [7, 7], padding='SAME', scope='conv4_9x9')
            large_column = relu(large_column)

        with tf.variable_scope('medium'):
            medium_column = slim.conv2d(inputs, 20, [7, 7], padding='SAME', scope='conv1_7x7')
            medium_column = relu(medium_column)
            medium_column = slim.conv2d(medium_column, 40, [5, 5], padding='SAME', scope='conv2_7x7')
            medium_column = relu(medium_column)
            medium_column = slim.max_pool2d(medium_column, [2, 2], 2, scope='pool1_7x7')
            medium_column = slim.conv2d(medium_column, 20, [5, 5], padding='SAME', scope='conv3_7x7')
            medium_column = relu(medium_column)
            medium_column = slim.max_pool2d(medium_column, [2, 2], 2, scope='pool2_7x7')
            medium_column = slim.conv2d(medium_column, 10, [5, 5], padding='SAME', scope='conv4_7x7')
            medium_column = relu(medium_column)

        with tf.variable_scope('small'):
            small_column = slim.conv2d(inputs, 24, [5, 5], padding='SAME', scope='conv1_5x5')
            small_column = relu(small_column)
            small_column = slim.conv2d(small_column, 48, [3, 3], padding='SAME', scope='conv2_5x5')
            small_column = relu(small_column)
            small_column = slim.max_pool2d(small_column, [2, 2], 2, scope='pool1_5x5')
            small_column = slim.conv2d(small_column, 24, [3, 3], padding='SAME', scope='conv3_5x5')
            small_column = relu(small_column)
            small_column = slim.max_pool2d(small_column, [2, 2], 2, scope='pool2_5x5')
            small_column = slim.conv2d(small_column, 12, [3, 3], padding='SAME', scope='conv4_5x5')
            small_column = relu(small_column)

        net = tf.concat([large_column, medium_column, small_column], axis=3)
        dmp = slim.conv2d(net, 1, [1, 1], padding='SAME', scope='dmp_conv1')

    return dmp

# network test
'''
import numpy as np

if __name__ == "__main__":
    x = tf.placeholder(tf.float32, [1, 700, 800, 1])
    net = build(x)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    d_map = sess.run(net,feed_dict={x:255*np.ones(shape=(1,700,800,1), dtype=np.float32)})
    prediction = np.asarray(d_map)
    print(prediction.shape)
    prediction = np.squeeze(prediction, axis=0)
    prediction = np.squeeze(prediction, axis=2)
'''

'''
def net9(x):
    # w x h x 3 -> w x h x 16
    net = conv(x, 16, kernel=9, stride=1, padding='same', scope='conv_1_9x9')
    net = relu(net)
    # w x h x 16 -> w/2 x h/2 x 16
    #net = pool(net, kh=2, kw=2, stride=2,  padding='same', scope='pool_1_9x9')
    # w/2 x h/2 x 16 -> w/2 x h/2 x 32
    net = conv(net,32, kernel=7, stride=1, padding='same', scope='conv_2_9x9')
    net = relu(net)
    # w/2 x h/2 x 32 -> w/4 x h/4 x 32
    net = pool(net, kh=2, kw=2, stride=2,  padding='same', scope='pool_2_9x9')
    # w/4 x h/4 x 32 -> w/4 x h/4 x 16
    net = conv(net, 16, kernel=7, stride=1, padding='same',scope='conv_3_9x9')
    net = relu(net)
    net = pool(net, kh=2, kw=2, stride=2, padding='same', scope='pool_1_9x9')
    # w/4 x h/4 x 32 -> w/4 x h/4 x 8
    net = conv(net, 8, kernel=7, stride=1, padding='same',scope='conv_4_9x9')
    net = relu(net)
    return net

def net7(x):
    # w x h x 3 -> w x h x 20
    net = conv(x, 20, kernel=7, stride=1, padding='same', scope='conv_1_7x7')
    net = relu(net)
    # w x h x 20 -> w/2 x h/2 x 20
    #net = pool(net, kh=2, kw=2, stride=2, padding='same', scope='pool_1_7x7')
    # w/2 x h/2 x 20 -> w/2 x h/2 x 40
    net = conv(net, 40, kernel=5, stride=1, padding='same', scope='conv_2_7x7')
    net = relu(net)
    # w/2 x h/2 x 40 -> w/4 x h/4 x 40
    net = pool(net, kh=2, kw=2, stride=2, padding='same', scope='pool_2_7x7')
    # w/4 x h/4 x 40 -> w/4 x h/4 x 20
    net = conv(net, 20, kernel=5, stride=1, padding='same', scope='conv_3_7x7')
    net = relu(net)
    net = pool(net, kh=2, kw=2, stride=2, padding='same', scope='pool_1_7x7')
    # w/4 x h/4 x 20 -> w/4 x h/4 x 10
    net = conv(net, 10, kernel=5, stride=1, padding='same', scope='conv_4_7x7')
    net = relu(net)
    return net

def net5(x):
    # w x h x 3 -> w x h x 24
    net = conv(x, 24, kernel=5, stride=1, padding='same', scope='conv_1_5x5')
    net = relu(net)
    # w x h x 24 -> w/2 x h/2 x 24
    # net = pool(net, kh=2, kw=2, stride=2, padding='same', scope='pool_1_5x5')
    # w/2 x h/2 x 24 -> w/2 x h/2 x 48
    net = conv(net, 48, kernel=3, stride=1, padding='same', scope='conv_2_5x5')
    net = relu(net)
    # w/2 x h/2 x 48 -> w/4 x h/4 x 48
    net = pool(net, kh=2, kw=2, stride=2, padding='same', scope='pool_2_5x5')
    # w/4 x h/4 x 48 -> w/4 x h/4 x 24
    net = conv(net, 24, kernel=3, stride=1, padding='same', scope='conv_3_5x5')
    net = relu(net)
    net = pool(net, kh=2, kw=2, stride=2, padding='same', scope='pool_1_5x5')
    # w/4 x h/4 x 24 -> w/4 x h/4 x 12
    net = conv(net, 12, kernel=3, stride=1, padding='same', scope='conv_4_5x5')
    net = relu(net)
    return net

def merge_net(net5,net7,net9):
    net = tf.concat([net5,net7,net9],axis=3)
    net = conv(net,1,kernel=1,stride=1,padding='same',scope='conv_merge')
    return net


def build(input_tensor, norm = False):
    """
    Builds the entire multi column cnn with 3 shallow nets with different input kernels and one fusing layer.
    :param input_tensor: Input tensor image to the network.
    :return: estimated density map tensor.
    """
    tf.summary.image('input', input_tensor, 1)
    if norm:
        input_tensor = tf.cast(input_tensor, tf.float32) * (1. / 255) - 0.5
    net_1_output = net9(input_tensor)                # For column 1 with large receptive fields
    net_2_output = net7(input_tensor)                # For column 2 with medium receptive fields
    net_3_output = net5(input_tensor)                # For column 3 with small receptive fields
    full_net = merge_net(net_1_output, net_2_output, net_3_output) # Fusing all the column output features
    return full_net
'''