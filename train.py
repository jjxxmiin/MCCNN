from utils import load,save
import tensorflow as tf
import mcnn as model
from ops import mse

'''


def save(checkpoint_dir, model_dir, step, sess, saver):
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess, os.path.join(checkpoint_dir, '.ckpt'), global_step=step)

saver = tf.train.Saver()

checkpoint_dir = os.path.join(checkpoint_dir,'model')

save(checkpoint_dir, counter, sess, saver)
'''

dataset_dir = "G:/dataset/people dataset/ShanghaiTech/part_A/"
log_dir = "logs"
checkpoint_dir = "checkpoint"
sample_dir = "sample"

learning_rate = 0.01

image = tf.placeholder(tf.float32,shape=[1,None,None,1])
ground_truth = tf.placeholder(tf.float32,shape=[1,None,None,1])

net9 = model.net9(image)
net7 = model.net7(image)
net5 = model.net5(image)

estimate = model.merge_net(net5,net7,net9)

# MSE
loss = mse(estimate,ground_truth)
# SGD : batch size 1
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

train = optimizer.minimize(loss)

tf.summary.scalar("SGD",optimizer)
tf.summary.scalar("Train",train)
tf.summary.scalar("MSE",loss)
summary = tf.summary.merge_all()

saver = tf.train.Saver()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # tensorboard
    writer = tf.summary.FileWriter(log_dir, sess.graph)

    # graph
    tf.train.write_graph(sess.graph_def, '.', 'graph.pbtxt')

    # model loading
    could_load, checkpoint_counter = load(checkpoint_dir, sess, saver)








