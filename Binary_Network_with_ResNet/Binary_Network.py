import _init_
from DeepLearning.deep_tensorflow import *


# 定义卷积，卷积后尺寸不变
def conv(img, weight, offset, scale, strides=1):
    conv_img = Bin_conv(img, weight, strides=strides)
    mean, variance = tf.nn.moments(conv_img, [0, 1, 2])
    conv_batch = tf.nn.batch_normalization(conv_img, mean, variance, offset, scale, 1e-10)
    return hard_tanh(conv_batch)


def Residual_Block(img, pre_img, weight, offset, scale, strides=1):
    conv_img = Bin_conv(img, weight, strides=strides)
    input_shape = pre_img.get_shape()[3]    # [?, , , 64]
    output_shape = conv_img.get_shape()[3]  # [?, , , 256]
    if input_shape != output_shape:
        weight_pre = tf.get_variable('weight_pre', [1, 1, input_shape, output_shape], initializer=tf.random_normal_initializer(mean=0, stddev=1))
        conv_pre_img = Bin_conv(pre_img, weight_pre, strides=strides)
        output = conv_img + conv_pre_img
        _init_.parameters += [weight_pre]
    else:
        output = conv_img + pre_img
    mean, variance = tf.nn.moments(output, [0, 1, 2])
    conv_batch = tf.nn.batch_normalization(output, mean, variance, offset, scale, 1e-10)
    return hard_tanh(conv_batch)


def train_loss(prediction, labels):
    prediction = tf.nn.softmax(prediction)
    cross_entropy = -tf.reduce_sum(labels * tf.log(prediction))        # 求和
    train_step = tf.train.GradientDescentOptimizer(_init_.learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return train_step, accuracy

"""
  in: [84, 84, 3]
  conv1 :[21, 21, 64]
  conv2: [10, 10, 256]
  conv3: [5, 5, 512]
  conv4: [2, 2, 1024]
  conv5: [1, 1, 2048]
  fc: []
"""


class ResNet:
    def __init__(self):
        self.img = None
        self.reuse = False
        self.learning_rate = _init_.learning_rate

    def __call__(self, img, scope):
        self.parameters = []
        self.img = img
        with tf.variable_scope(scope, reuse=self.reuse) as scope_name:
            if self.reuse:
                scope_name.reuse_variables()
            # conv1
            with tf.variable_scope('conv1'):
                weight_1 = tf.get_variable('weight', shape=[7, 7, 3, 64], initializer=tf.random_normal_initializer(mean=0, stddev=1))   # [k_size, k_size, input_size, output_size]
                offset_1 = tf.get_variable('offset', [64], initializer=tf.constant_initializer(0.0))
                scale_1 = tf.get_variable('scale', [64], initializer=tf.constant_initializer(1.0))
                conv1_ReLu = conv(self.img, weight_1,  offset_1, scale_1, strides=2)
                conv1 = ave_pool(conv1_ReLu, k_size=(2, 2), stride=(2, 2))
                _init_.parameters += [weight_1, offset_1, scale_1]
            # conv2
            with tf.variable_scope('conv2'):
                with tf.variable_scope('conv2_1'):
                    with tf.variable_scope('conv2_1_1'):
                        weight2_1_1 = tf.get_variable('weight', shape=[1, 1, 64, 64], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        offset2_1_1 = tf.get_variable('offset', [64], initializer=tf.constant_initializer(0.0))
                        scale2_1_1 = tf.get_variable('scale', [64], initializer=tf.constant_initializer(1.0))
                        conv2_1_1_ReLu = conv(conv1, weight2_1_1, offset2_1_1, scale2_1_1, strides=1)
                        _init_.parameters += [weight2_1_1,  offset2_1_1, scale2_1_1]
                    with tf.variable_scope('conv2_1_2'):
                        weight2_1_2 = tf.get_variable('weight', [3, 3, 64, 64], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        offset2_1_2 = tf.get_variable('offset', [64], initializer=tf.constant_initializer(0.0))
                        scale2_1_2 = tf.get_variable('scale', [64], initializer=tf.constant_initializer(1.0))
                        conv2_1_2_ReLu = conv(conv2_1_1_ReLu, weight2_1_2, offset2_1_2, scale2_1_2, strides=1)
                        _init_.parameters += [weight2_1_2, offset2_1_2, scale2_1_2]
                    with tf.variable_scope('conv2_1_3'):
                        weight2_1_3 = tf.get_variable('weight', [1, 1, 64, 256], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        offset2_1_3 = tf.get_variable('offset', [256], initializer=tf.constant_initializer(0.0))
                        scale2_1_3 = tf.get_variable('scale', [256], initializer=tf.constant_initializer(1.0))
                        conv2_1 = Residual_Block(conv2_1_2_ReLu, conv1,  weight2_1_3, offset2_1_3,
                                                        scale2_1_3,  strides=1)
                        _init_.parameters += [weight2_1_3, offset2_1_3, scale2_1_3]
                with tf.variable_scope('conv2_2'):
                    with tf.variable_scope('conv2_2_1'):
                        weight2_2_1 = tf.get_variable('weight', [1, 1, 256, 64], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        offset2_2_1 = tf.get_variable('offset', [64], initializer=tf.constant_initializer(0.0))
                        scale2_2_1 = tf.get_variable('scale', [64], initializer=tf.constant_initializer(1.0))
                        conv2_2_1_ReLu = conv(conv2_1, weight2_2_1,  offset2_2_1, scale2_2_1, strides=1)
                        _init_.parameters += [weight2_2_1,  offset2_2_1, scale2_2_1]
                    with tf.variable_scope('conv2_2_2'):
                        weight2_2_2 = tf.get_variable('weight', [3, 3, 64, 64], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        offset2_2_2 = tf.get_variable('offset', [64], initializer=tf.constant_initializer(0.0))
                        scale2_2_2 = tf.get_variable('scale', [64], initializer=tf.constant_initializer(1.0))
                        conv2_2_2_ReLu = conv(conv2_2_1_ReLu, weight2_2_2, offset2_2_2, scale2_2_2, strides=1)
                        _init_.parameters += [weight2_2_2,  offset2_2_2, scale2_2_2]
                    with tf.variable_scope('conv2_2_3'):
                        weight2_2_3 = tf.get_variable('weight', [1, 1, 64, 256], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        offset2_2_3 = tf.get_variable('offset', [256], initializer=tf.constant_initializer(0.0))
                        scale2_2_3 = tf.get_variable('scale', [256], initializer=tf.constant_initializer(1.0))
                        conv2_2 = Residual_Block(conv2_2_2_ReLu, conv2_1,  weight2_2_3,  offset2_2_3,
                                                        scale2_2_3,  strides=1)
                        _init_.parameters += [weight2_2_3,  offset2_2_3, scale2_2_3]
                with tf.variable_scope('conv2_3'):
                    with tf.variable_scope('conv2_3_1'):
                        weight2_3_1 = tf.get_variable('weight', [1, 1, 256, 64], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        offset2_3_1 = tf.get_variable('offset', [64], initializer=tf.constant_initializer(0.0))
                        scale2_3_1 = tf.get_variable('scale', [64], initializer=tf.constant_initializer(1.0))
                        conv2_3_1_ReLu = conv(conv2_2, weight2_3_1,  offset2_3_1, scale2_3_1, strides=1)
                        _init_.parameters += [weight2_3_1, offset2_3_1, scale2_3_1]
                    with tf.variable_scope('conv2_3_2'):
                        weight2_3_2 = tf.get_variable('weight', [3, 3, 64, 64], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        offset2_3_2 = tf.get_variable('offset', [64], initializer=tf.constant_initializer(0.0))
                        scale2_3_2 = tf.get_variable('scale', [64], initializer=tf.constant_initializer(1.0))
                        conv2_3_2_ReLu = conv(conv2_3_1_ReLu, weight2_3_2, offset2_3_2, scale2_3_2, strides=1)
                        _init_.parameters += [weight2_3_2,  offset2_3_2, scale2_3_2]
                    with tf.variable_scope('conv2_3_3'):
                        weight2_3_3 = tf.get_variable('weight', [1, 1, 64, 256], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        offset2_3_3 = tf.get_variable('offset', [256], initializer=tf.constant_initializer(0.0))
                        scale2_3_3 = tf.get_variable('scale', [256], initializer=tf.constant_initializer(1.0))
                        conv2_3 = Residual_Block(conv2_3_2_ReLu, conv2_2,  weight2_3_3,  offset2_3_3,
                                                        scale2_3_3,  strides=1)
                        _init_.parameters += [weight2_3_3,  offset2_3_3, scale2_3_3]
                        conv2 = ave_pool(conv2_3, k_size=(2, 2), stride=(2, 2))
            # conv3
            with tf.variable_scope('conv3'):
                with tf.variable_scope('conv3_1'):
                    with tf.variable_scope('conv3_1_1'):
                        weight3_1_1 = tf.get_variable('weight', [1, 1, 256, 128], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        offset3_1_1 = tf.get_variable('offset', [128], initializer=tf.constant_initializer(0.0))
                        scale3_1_1 = tf.get_variable('scale', [128], initializer=tf.constant_initializer(1.0))
                        conv3_1_1_ReLu = conv(conv2, weight3_1_1, offset3_1_1, scale3_1_1, strides=1)
                        _init_.parameters += [weight3_1_1,  offset3_1_1, scale3_1_1]
                    with tf.variable_scope('conv3_1_2'):
                        weight3_1_2 = tf.get_variable('weight', [3, 3, 128, 128], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        offset3_1_2 = tf.get_variable('offset', [128], initializer=tf.constant_initializer(0.0))
                        scale3_1_2 = tf.get_variable('scale', [128], initializer=tf.constant_initializer(1.0))
                        conv3_1_2_ReLu = conv(conv3_1_1_ReLu, weight3_1_2, offset3_1_2, scale3_1_2, strides=1)
                        _init_.parameters += [weight3_1_2, offset3_1_2, scale3_1_2]
                    with tf.variable_scope('conv3_1_3'):
                        weight3_1_3 = tf.get_variable('weight', [1, 1, 128, 512], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        offset3_1_3 = tf.get_variable('offset', [512], initializer=tf.constant_initializer(0.0))
                        scale3_1_3 = tf.get_variable('scale', [512], initializer=tf.constant_initializer(1.0))
                        conv3_1 = Residual_Block(conv3_1_2_ReLu, conv2,  weight3_1_3,  offset3_1_3,
                                                        scale3_1_3,  strides=1)
                        _init_.parameters += [weight3_1_3,  offset3_1_3, scale3_1_3]
                with tf.variable_scope('conv3_2'):
                    with tf.variable_scope('conv3_2_1'):
                        weight3_2_1 = tf.get_variable('weight', [1, 1, 512, 128], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        offset3_2_1 = tf.get_variable('offset', [128], initializer=tf.constant_initializer(0.0))
                        scale3_2_1 = tf.get_variable('scale', [128], initializer=tf.constant_initializer(1.0))
                        conv3_2_1_ReLu = conv(conv3_1, weight3_2_1,  offset3_2_1, scale3_2_1, strides=1)
                        _init_.parameters += [weight3_2_1,  offset3_2_1, scale3_2_1]
                    with tf.variable_scope('conv3_2_2'):
                        weight3_2_2 = tf.get_variable('weight', [3, 3, 128, 128], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        offset3_2_2 = tf.get_variable('offset', [128], initializer=tf.constant_initializer(0.0))
                        scale3_2_2 = tf.get_variable('scale', [128], initializer=tf.constant_initializer(1.0))
                        conv3_2_2_ReLu = conv(conv3_2_1_ReLu, weight3_2_2,  offset3_2_2, scale3_2_2, strides=1)
                        _init_.parameters += [weight3_2_2, offset3_2_2, scale3_2_2]
                    with tf.variable_scope('conv3_2_3'):
                        weight3_2_3 = tf.get_variable('weight', [1, 1, 128, 512], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        offset3_2_3 = tf.get_variable('offset', [512], initializer=tf.constant_initializer(0.0))
                        scale3_2_3 = tf.get_variable('scale', [512], initializer=tf.constant_initializer(1.0))
                        conv3_2 = Residual_Block(conv3_2_2_ReLu, conv3_1,  weight3_2_3,  offset3_2_3,
                                                        scale3_2_3,  strides=1)
                        _init_.parameters += [weight3_2_3,  offset3_2_3, scale3_2_3]
                with tf.variable_scope('conv3_3'):
                    with tf.variable_scope('conv3_3_1'):
                        weight3_3_1 = tf.get_variable('weight', [1, 1, 512, 128], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        offset3_3_1 = tf.get_variable('offset', [128], initializer=tf.constant_initializer(0.0))
                        scale3_3_1 = tf.get_variable('scale', [128], initializer=tf.constant_initializer(1.0))
                        conv3_3_1_ReLu = conv(conv3_2, weight3_3_1,  offset3_3_1, scale3_3_1, strides=1)
                        _init_.parameters += [weight3_3_1,  offset3_3_1, scale3_3_1]
                    with tf.variable_scope('conv3_3_2'):
                        weight3_3_2 = tf.get_variable('weight', [3, 3, 128, 128], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        offset3_3_2 = tf.get_variable('offset', [128], initializer=tf.constant_initializer(0.0))
                        scale3_3_2 = tf.get_variable('scale', [128], initializer=tf.constant_initializer(1.0))
                        conv3_3_2_ReLu = conv(conv3_3_1_ReLu, weight3_3_2, offset3_3_2, scale3_3_2, strides=1)
                        _init_.parameters += [weight3_3_2,  offset3_3_2, scale3_3_2]
                    with tf.variable_scope('conv3_3_3'):
                        weight3_3_3 = tf.get_variable('weight', [1, 1, 128, 512], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        offset3_3_3 = tf.get_variable('offset', [512], initializer=tf.constant_initializer(0.0))
                        scale3_3_3 = tf.get_variable('scale', [512], initializer=tf.constant_initializer(1.0))
                        conv3_3 = Residual_Block(conv3_3_2_ReLu, conv3_2,  weight3_3_3, offset3_3_3,
                                                        scale3_3_3,  strides=1)
                        _init_.parameters += [weight3_3_3,  offset3_3_3, scale3_3_3]
                with tf.variable_scope('conv3_4'):
                    with tf.variable_scope('conv3_4_1'):
                        weight3_4_1 = tf.get_variable('weight', [1, 1, 512, 128], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        offset3_4_1 = tf.get_variable('offset', [128], initializer=tf.constant_initializer(0.0))
                        scale3_4_1 = tf.get_variable('scale', [128], initializer=tf.constant_initializer(1.0))
                        conv3_4_1_ReLu = conv(conv3_3, weight3_4_1, offset3_4_1, scale3_4_1,
                                              strides=1)
                        _init_.parameters += [weight3_4_1,  offset3_4_1, scale3_4_1]
                    with tf.variable_scope('conv3_4_2'):
                        weight3_4_2 = tf.get_variable('weight', [3, 3, 128, 128], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        offset3_4_2 = tf.get_variable('offset', [128], initializer=tf.constant_initializer(0.0))
                        scale3_4_2 = tf.get_variable('scale', [128], initializer=tf.constant_initializer(1.0))
                        conv3_4_2_ReLu = conv(conv3_4_1_ReLu, weight3_4_2, offset3_4_2, scale3_4_2,
                                              strides=1)
                        _init_.parameters += [weight3_4_2,  offset3_4_2, scale3_4_2]
                    with tf.variable_scope('conv3_4_3'):
                        weight3_4_3 = tf.get_variable('weight', [1, 1, 128, 512], initializer=tf.random_normal_initializer(mean=0, stddev=1))
                        offset3_4_3 = tf.get_variable('offset', [512], initializer=tf.constant_initializer(0.0))
                        scale3_4_3 = tf.get_variable('scale', [512], initializer=tf.constant_initializer(1.0))
                        conv3_4 = Residual_Block(conv3_4_2_ReLu, conv3_3, weight3_4_3,  offset3_4_3,
                                                 scale3_4_3, strides=1)
                        _init_.parameters += [weight3_4_3,  offset3_4_3, scale3_4_3]

                        conv3 = ave_pool(conv3_4, k_size=(2, 2), stride=(2, 2))
            # conv4
            in_img = conv3
            with tf.variable_scope('conv4'):
                for kk in range(23):
                    with tf.variable_scope('conv4_' + str(kk)):
                        with tf.variable_scope('conv4_1'):
                            in_shape = in_img.get_shape()[3]
                            weight4_1 = tf.get_variable('weight', [1, 1, in_shape, 256])
                            offset4_1 = tf.get_variable('offset', [256], initializer=tf.constant_initializer(0.0))
                            scale4_1 = tf.get_variable('scale', [256], initializer=tf.constant_initializer(1.0))
                            conv4_1_ReLu = conv(in_img, weight4_1, offset4_1, scale4_1,
                                                  strides=1)
                            _init_.parameters += [weight4_1,  offset4_1, scale4_1]
                        with tf.variable_scope('conv4_2'):
                            weight4_2 = tf.get_variable('weight', [3, 3, 256, 256])
                            offset4_2 = tf.get_variable('offset', [256], initializer=tf.constant_initializer(0.0))
                            scale4_2 = tf.get_variable('scale', [256], initializer=tf.constant_initializer(1.0))
                            conv4_2_ReLu = conv(conv4_1_ReLu, weight4_2, offset4_2, scale4_2,
                                                  strides=1)
                            _init_.parameters += [weight4_2, offset4_2, scale4_2]
                        with tf.variable_scope('conv4_3'):
                            weight4_3 = tf.get_variable('weight', [1, 1, 256, 1024])
                            offset4_3 = tf.get_variable('offset', [1024], initializer=tf.constant_initializer(0.0))
                            scale4_3 = tf.get_variable('scale', [1024], initializer=tf.constant_initializer(1.0))
                            conv3_4 = Residual_Block(conv4_2_ReLu, in_img, weight4_3,  offset4_3,
                                                     scale4_3, strides=1)
                            _init_.parameters += [weight4_3,  offset4_3, scale4_3]
                    in_img = conv3_4
                conv4 = ave_pool(in_img, k_size=(2, 2), stride=(2, 2))
            # conv5
            with tf.variable_scope('conv5'):
                with tf.variable_scope('conv5_1'):
                    with tf.variable_scope('conv5_1_1'):
                        weight5_1_1 = tf.get_variable('weight', [1, 1, 1024, 512])
                        offset5_1_1 = tf.get_variable('offset', [512], initializer=tf.constant_initializer(0.0))
                        scale5_1_1 = tf.get_variable('scale', [512], initializer=tf.constant_initializer(1.0))
                        conv5_1_1_ReLu = conv(conv4, weight5_1_1,  offset5_1_1, scale5_1_1, strides=1)
                        _init_.parameters += [weight5_1_1,  offset5_1_1, scale5_1_1]
                    with tf.variable_scope('conv5_1_2'):
                        weight5_1_2 = tf.get_variable('weight', [3, 3, 512, 512])
                        offset5_1_2 = tf.get_variable('offset', [512], initializer=tf.constant_initializer(0.0))
                        scale5_1_2 = tf.get_variable('scale', [512], initializer=tf.constant_initializer(1.0))
                        conv5_1_2_ReLu = conv(conv5_1_1_ReLu, weight5_1_2, offset5_1_2, scale5_1_2,
                                              strides=1)
                        _init_.parameters += [weight5_1_2, offset5_1_2, scale5_1_2]
                    with tf.variable_scope('conv5_1_3'):
                        weight5_1_3 = tf.get_variable('weight', [1, 1, 512, 2048])
                        offset5_1_3 = tf.get_variable('offset', [2048], initializer=tf.constant_initializer(0.0))
                        scale5_1_3 = tf.get_variable('scale', [2048], initializer=tf.constant_initializer(1.0))
                        conv5_1 = Residual_Block(conv5_1_2_ReLu, conv4, weight5_1_3,  offset5_1_3,
                                                 scale5_1_3, strides=1)
                        _init_.parameters += [weight5_1_3, offset5_1_3, scale5_1_3]
                with tf.variable_scope('conv5_2'):
                    with tf.variable_scope('conv5_2_1'):
                        weight5_2_1 = tf.get_variable('weight', [1, 1, 2048, 512])
                        offset5_2_1 = tf.get_variable('offset', [512], initializer=tf.constant_initializer(0.0))
                        scale5_2_1 = tf.get_variable('scale', [512], initializer=tf.constant_initializer(1.0))
                        conv5_2_1_ReLu = conv(conv5_1, weight5_2_1,  offset5_2_1, scale5_2_1, strides=1)
                        _init_.parameters += [weight5_2_1,  offset5_2_1, scale5_2_1]
                    with tf.variable_scope('conv5_2_2'):
                        weight5_2_2 = tf.get_variable('weight', [3, 3, 512, 512])
                        offset5_2_2 = tf.get_variable('offset', [512], initializer=tf.constant_initializer(0.0))
                        scale5_2_2 = tf.get_variable('scale', [512], initializer=tf.constant_initializer(1.0))
                        conv5_2_2_ReLu = conv(conv5_2_1_ReLu, weight5_2_2,  offset5_2_2, scale5_2_2,
                                              strides=1)
                        _init_.parameters += [weight5_2_2,  offset5_2_2, scale5_2_2]
                    with tf.variable_scope('conv5_2_3'):
                        weight5_2_3 = tf.get_variable('weight', [1, 1, 512, 2048])
                        offset5_2_3 = tf.get_variable('offset', [2048], initializer=tf.constant_initializer(0.0))
                        scale5_2_3 = tf.get_variable('scale', [2048], initializer=tf.constant_initializer(1.0))
                        conv5_2 = Residual_Block(conv5_2_2_ReLu, conv5_1, weight5_2_3,  offset5_2_3,
                                                 scale5_2_3, strides=1)
                        _init_.parameters += [weight5_2_3, offset5_2_3, scale5_2_3]
                with tf.variable_scope('conv5_3'):
                    with tf.variable_scope('conv5_3_1'):
                        weight5_3_1 = tf.get_variable('weight', [1, 1, 2048, 512])
                        offset5_3_1 = tf.get_variable('offset', [512], initializer=tf.constant_initializer(0.0))
                        scale5_3_1 = tf.get_variable('scale', [512], initializer=tf.constant_initializer(1.0))
                        conv5_3_1_ReLu = conv(conv5_2, weight5_3_1,  offset5_3_1, scale5_3_1, strides=1)
                        _init_.parameters += [weight5_3_1,  offset5_3_1, scale5_3_1]
                    with tf.variable_scope('conv5_3_2'):
                        weight5_3_2 = tf.get_variable('weight', [3, 3, 512, 512])
                        offset5_3_2 = tf.get_variable('offset', [512], initializer=tf.constant_initializer(0.0))
                        scale5_3_2 = tf.get_variable('scale', [512], initializer=tf.constant_initializer(1.0))
                        conv5_3_2_ReLu = conv(conv5_3_1_ReLu, weight5_3_2, offset5_3_2, scale5_3_2,
                                              strides=1)
                        _init_.parameters += [weight5_3_2, offset5_3_2, scale5_3_2]
                    with tf.variable_scope('conv2_3_3'):
                        weight5_3_3 = tf.get_variable('weight', [1, 1, 512, 2048])
                        offset5_3_3 = tf.get_variable('offset', [2048], initializer=tf.constant_initializer(0.0))
                        scale5_3_3 = tf.get_variable('scale', [2048], initializer=tf.constant_initializer(1.0))
                        conv5_3 = Residual_Block(conv5_3_2_ReLu, conv5_2, weight5_3_3, offset5_3_3,
                                                 scale5_3_3, strides=1)
                        _init_.parameters += [weight5_3_3,  offset5_3_3, scale5_3_3]
                        conv5 = ave_pool(conv5_3, k_size=(2, 2), stride=(2, 2))  # [1, 1, 2048]
            # fc
            fc_in = tf.squeeze(conv5, [1, 2])  # [batch_size, 2048]
            with tf.variable_scope('fc'):
                weight_fc2 = tf.get_variable('weight', [2048, _init_.classes_numbers])
                B_fc = tf.sign(weight_fc2)
                alpha_fc = tf.reduce_sum(abs(weight_fc2), 0)/2048*_init_.classes_numbers
                fc = tf.multiply(tf.matmul(fc_in, B_fc), alpha_fc)
                fc = tf.div(fc, 10)  # 或者加个BN使得输出fc小一点，这样lr可以大点
                _init_.parameters += [weight_fc2]
        self.reuse = True
        return fc, conv1





