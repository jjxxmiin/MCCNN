# GAN OPS
# Loss,Activation,Normalization,Layer

import tensorflow as tf
import tensorflow.contrib as tf_contrib

weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
# weight_init = tf.truncated_normal(mean=0.0,stddev=0.2)
weight_regularizer = None

# ===================================== layer =================================================
def conv(x, channels, kernel, stride, padding='vaild', use_bias=True, scope='conv'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x,
                             filters=channels,
                             kernel_size=kernel,
                             padding=padding,
                             kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias)

        return x

def deconv(x, channels, kernel=3, stride=2, use_bias=True, scope='deconv_0') :
    with tf.variable_scope(scope):
        x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                       kernel_size=kernel, kernel_initializer=weight_init,
                                       kernel_regularizer=weight_regularizer,
                                       strides=stride, use_bias=use_bias, padding='SAME')

        return x

def pool(x,kw,kh,stride,padding='vaild',scope='pool_0'):
    with tf.variable_scope(scope):
        return tf.layers.max_pooling2d(x,pool_size=[kh,kw],strides=stride,padding=padding)

# ===================================== Activation ================================================
def lrelu(x, alpha=0.1):
    #tf.nn.leaky_relu(x,alpha=alpha)
    return tf.maximum(x, alpha*x)

def relu(x):
    #tf.nn.relu(x)
    return tf.maximum(x,0)

def sigmoid(x):
    # tf.nn.sigmoid(x)
    return 1 / (1+tf.exp(-x))

def tanh(x):
    return tf.tanh(x)

# Normalization
def instance_norm(x,scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True,
                                           scale=True,
                                           scope=scope)

def batch_norm(x,scope='batch_norm'):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-05,
                                        scale=True,
                                        is_training=True,
                                        scope=scope,
                                        reuse=tf.AUTO_REUSE  # if tensorflow vesrion < 1.4, delete this line
                                        )

# ===================================== Residual block ================================================
def resblock(x,filter,scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            y = conv(x,filter,kernel=3,stride=1,pad=1,pad_type='reflect',use_bias=True)
            y = instance_norm(x)
            y = relu(x)

        with tf.variable_scope('res2'):
            y = conv(y,filter,kernel=3,stride=1,pad=1,pad_type='reflect',use_bias=True)
            y = instance_norm(x)

        return x + y

# ===================================== Loss ================================================
def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss

def L2_loss(x, y):
    loss = tf.reduce_mean(tf.pow((x - y),2))

    return loss

# MSE(Mean Squared Error) : (오차)값이 작을수록  정답에 가깝다.
# yi는 신경망의 출력, ti는 정답 레이블(One-Hot인코딩)

# Y : 추정 density map
# T : 실제 density map
def mse(y,t):
    loss = 0.5 * tf.reduce_sum(tf.square(tf.subtract(y, t)))

    return loss