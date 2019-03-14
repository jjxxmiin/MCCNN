from utils import *
import tensorflow as tf
import mcnn as model
from ops import mse
import matplotlib.image as mpimg
import scipy.io as sio

dataset_dir = "G:\dataset\people dataset\ShanghaiTech\part_B"
log_dir = "logs"
checkpoint_dir = "checkpoint"
sample_dir = "sample"

learning_rate = 0.001
epoch = 200

image = tf.placeholder(tf.float32,shape=[1,None,None,3])
ground_truth = tf.placeholder(tf.float32,shape=[1,None,None,1])

'''
net9 = model.net9(image)
net7 = model.net7(image)
net5 = model.net5(image)

estimate = model.merge_net(net5,net7,net9)
'''
estimate = model.multi_column_cnn(image)

# MSE
loss = mse(estimate,ground_truth)
# SGD : batch size 1
optimizer = tf.train.AdamOptimizer(learning_rate)

train = optimizer.minimize(loss)

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

    if could_load:
        counter = checkpoint_counter
        print(" [*] Load SUCCESS")
    else:
        counter = 1
        print(" [!] Load failed...")

    for e in range(epoch):
        train_image_list, train_gt_list, iteration = get_data_list(dataset_dir,mode='train')

        for i in range(iteration):
            img,gt_dmp,gt_count = read_test_data(train_image_list[i],train_gt_list[i],scale=4)
            #print(img,gt_dmp)
            input_normalization(img)

            _,prediction,cost,summary_str = sess.run([train,estimate,loss,summary],feed_dict={image : img ,
                                                       ground_truth : gt_dmp
                                                       })
            writer.add_summary(summary_str, counter)
            counter += 1

            print("[{} / {}] [{} / {}] LOSS : {}".format(epoch,e,iteration,i,cost))

            absolute_error = np.abs(np.subtract(gt_count, prediction.sum())).mean()
            square_error = np.power(np.subtract(gt_count, prediction.sum()), 2).mean()

            mae = absolute_error / iteration
            rmse = np.sqrt(absolute_error / iteration)

            print(str('MAE : ' + str(mae) + ' MSE : ' + str(rmse)))


        save(checkpoint_dir, "mcnn", counter, sess, saver)







