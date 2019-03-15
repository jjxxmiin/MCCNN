from utils import *
import tensorflow as tf
import mcnn as model
from ops import mse
import matplotlib.image as mpimg
import scipy.io as sio

dataset = 'A'

log_dir = "logs"
checkpoint_dir = "checkpoint"
sample_dir = "sample"

learning_rate = 1e-6
epoch = 200000

image = tf.placeholder(tf.float32,shape=[1,None,None,3])
ground_truth = tf.placeholder(tf.float32,shape=[1,None,None,1])

estimate = model.multi_column_cnn_relu(image)

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

    # model loading
    could_load, checkpoint_counter = load(checkpoint_dir, sess, saver)

    train_image_list, train_gt_list, iteration = get_data_list(dataset, mode='train')

    if could_load:
        counter = checkpoint_counter
        start_epoch = checkpoint_counter // iteration
        print(" [*] Load SUCCESS")
    else:
        counter = 1
        start_epoch = 0
        print(" [!] Load failed...")

    for e in range(start_epoch,epoch):

        for i in range(iteration):
            img,gt_dmp,gt_count = read_train_data(train_image_list[i],train_gt_list[i],scale=4)

            img = input_normalization(img)

            _,prediction,cost,summary_str = sess.run([train,estimate,loss,summary],feed_dict={image : img ,
                                                       ground_truth : gt_dmp
                                                       })

            input_denormalization(prediction)

            writer.add_summary(summary_str, counter)
            counter += 1

            print("[{} / {}] [{} / {}] LOSS : {} pre : {} gt : {}".format(epoch,e,iteration,i,cost,prediction.sum(),gt_count))

        if e % 101 == 0:
            absolute_error = 0.0
            square_error = 0.0

            test_image_list, test_gt_list, total_test_count = get_data_list(dataset, mode='test')

            # validating
            for j in range(total_test_count):

                img, gt_dmp, gt_count = read_test_data(test_image_list[j], test_gt_list[j], scale=4)

                img = input_normalization(img)

                _, prediction, cost, summary_str = sess.run([train, estimate, loss, summary], feed_dict={image: img,
                                                                                                         ground_truth: gt_dmp
                                                                                                         })

                prediction = input_denormalization(prediction)

                absolute_error = absolute_error + np.abs(np.subtract(gt_count, prediction.sum())).mean()
                square_error = square_error + np.power(np.subtract(gt_count, prediction.sum()), 2).mean()

                print("[{} / {}] [{} / {}] LOSS : {} pre : {} gt : {}".format(epoch, e, total_test_count, j, cost,
                                                                              prediction.sum(), gt_count))

            mae = absolute_error / total_test_count
            rmse = np.sqrt(absolute_error / total_test_count)

            print(str('MAE_' + str(mae) + '_MSE_' + str(rmse)))

        if e % 100 == 0:
            save(checkpoint_dir, "mcnn", counter, sess, saver)
            # graph
            tf.train.write_graph(sess.graph_def, '.', 'graph.pbtxt')

        train_image_list, train_gt_list, _ = get_data_list(dataset,mode='train')






