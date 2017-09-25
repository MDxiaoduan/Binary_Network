# ------------------------------------------------------------------------ #
# XNOR Network with tensorflow                                             #
# Reference:XNOR-Net: ImageNet Classification Using Binary Convolutional   #
#                     Neural Networks                                      #
# python 3.5                                                               #
# tensorflow 1.1                                                           #
# numpy                                                                    #
# 2/9/2017                                                                 #
#                                                                          #
# Data: ImageNet                                                           #
# ------------------------------------------------------------------------ #
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from get_images import get_images
from DeepLearning.deep_learning import learning_rate
from Binary_Network import ResNet, train_loss

import _init_

cnn = ResNet()
data = get_images()
# tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, _init_.input_image[0], _init_.input_image[1], _init_.input_image[2]])
y = tf.placeholder(tf.float32, [None, _init_.classes_numbers])
# y = tf.placeholder(tf.int32, [None])   # y_5
fc_out, conv5 = cnn(x, scope='resnet')
train_step, acc_1 = train_loss(fc_out, y)


config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    Acc_1 = []
    for kk in range(1, _init_.iteration_numbers):
        img_batch, label_batch = data.get_mini_batch()
        # init_.learning_rate = learning_rate(kk*_init_.batch_size, "epochs", _init_.learning_rate, _init_.epochs_number*3)
        _, accuracy1, conv = sess.run([train_step, acc_1, conv5], feed_dict={x: img_batch, y: label_batch})
        # print("---------------------------------------------------------------------------------------------")
        # print(conv)
        if kk % _init_.display_step == 0:
            img_batch, label_batch = data.get_mini_batch(train=False)
            accuracy_1 = sess.run(acc_1, feed_dict={x: img_batch, y: label_batch})
            Acc_1.append(accuracy_1)
            print("Batch: ", kk, "Accuracy: ", accuracy_1)

    line1, = plt.plot(np.arange(len(Acc_1)), Acc_1, "c", markersize=10)
    # line2, = plt.plot(np.arange(len(Acc_5)), Acc_5, "c", markersize=10)
    plt.legend([line1], ["Top-1"])
    plt.xlabel('iter*1000')
    plt.ylabel('Acc')
    plt.title('Binary_Net_with_ResNet')
    plt.savefig("Binary_Net.jpg")  # 保存图片
    plt.show()
