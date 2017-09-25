# ------------------------------------------------------------------------ #
# Binary Network and XNOR Network                                          #
# Reference:XNOR-Net: ImageNet Classification Using Binary Convolutional   #
#                     Neural Networks                                      #
# python 3.5                                                               #
# numpy                                                                    #
# 26/8/2017                                                                #
# UESTC--Duan                                                              #
# Data: CIFAR10   or  mnist                                                #
# ------------------------------------------------------------------------ #
# import Lib
from __future__ import division        # for 1/2=0.5
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

import load_data

from DeepLearning.deep_learning import *
# convolve, sigmoid, ReLu, sigmoid_derivative, ReLu_derivative, Batch_Normalization, pooling, expand, deconvolution
"""
Lenet_model:
   input: 28*28
   convolution1:
         weight_1: size: [1, 6, 5, 5] （1个输入，6个输出，一共1*6个卷积核,每个卷积核大小5*5）
         bias_1: size: 6  6个输出所以只有6个bias(每个bias是一个数)
         output_size: 6*24*24   24=(28-5)/1+1
         pooling_1: size: [2, 2]  2*2大小做一个均值池化
         output_size: 6*12*12
   convolution2:
         weight_2: size: [6, 12, 5, 5] （6个输入，12个输出，一共6*12个卷积核,每个卷积核大小5*5）
         bias_2: size: 12
         output_size: 12*8*8   8=(12-5)/1+1
         pooling_2: size: [2, 2]  2*2大小做一个均值池化
         output_size: 12*4*4
   fully connected:
         input_size:12*4*4=192  --12个4*4输出变为一个向量
         weight_out: size: [192, 10]  (192个输入（每个输入就是一个数，10个分类输出）)
         bias_out: size: 10  10个输出
         output_size: 10
"""


class cnn_model:
    def __init__(self):
        """
        权重初始化方法：
            激活函数为sigmoid:
                 w = np.random.randn(n)/sqrt(n)        # n=k*k*c   k是输入大小，c是输入channel
            激活函数为ReLu:
                 w = np.random.randn(n)*sqrt(2.0/n)   # [Ref:https://arxiv.org/pdf/1502.01852.pdf]
            其他：
                w = np.random.randn(n)*sqrt(2.0/(n_in_size + n_out_size))
        """
        self.input_size = [28, 28]
        self.model_size = [1, 6, 12, 192, 10]     # [输入图片, 第一层卷积输出维数， 第二层卷积维数，  全链接输入， 分类标签数]
        self.weight_1 = np.random.randn(1, 6, 5, 5) * np.sqrt(2.0 / (1 + 6))
        self.bias_1 = np.random.randn(6)
        self.weight_2 = np.random.randn(6, 12, 5, 5) * np.sqrt(2.0 / (1 + 6))
        self.bias_2 = np.zeros(12)
        self.weight_out = np.random.randn(192, 10)
        self.bias_out = np.zeros(10)
        self.gamma_1 = np.ones(6)
        self.beta_1 = np.zeros(6)
        self.gamma_2 = np.ones(12)
        self.beta_2 = np.zeros(12)
        self.der_gamma_1 = np.zeros(6)
        self.der_beta_1 = np.zeros(6)
        self.der_gamma_2 = np.zeros(12)
        self.der_beta_2 = np.zeros(12)
        # self.batch_size = 1      # 多少张图片一起训练
        self.class_number = 10
        self.learning_rate = 0.001   # 学习速率
        self.iteration_number = 800000
        # 定义矩阵存储中间卷积后矩阵
        self.con1 = np.zeros((self.model_size[1], 24, 24))
        self.con1_out = np.zeros((self.model_size[1], 24, 24))
        self.con1_batch_out = np.zeros((self.model_size[1], 24, 24))
        self.pooling1_out = np.zeros((self.model_size[1], 12, 12))
        self.pooling2_out = np.zeros((self.model_size[2], 4, 4))
        self.full_con = np.zeros(192)
        self.t = np.zeros(self.class_number)
        self.a = np.zeros(self.class_number)
        self.d_out_w = np.zeros((192, 10))
        self.d_out_bias = np.zeros(10)
        self.der_full_con = np.zeros(192)
        self.der_pool_2 = np.zeros((12, 4, 4))
        self.der_con_2 = np.zeros((12, 8, 8))
        self.der_weight_2 = np.zeros((6, 12, 5, 5))
        self.d_con2_weight = np.zeros((6, 12, 5, 5))
        self.d_con2_bias = np.zeros(12)
        self.der_con_1 = np.zeros((6, 24, 24))
        self.der_weight_1 = np.zeros((1, 6, 5, 5))
        self.d_con1_weight = np.zeros((1, 6, 5, 5))
        self.d_con1_bias = np.zeros(6)
        self.der_out = np.zeros(10)
        self.con2_out = np.zeros((self.model_size[2], 8, 8))
        self.con2_batch_out = np.zeros((self.model_size[2], 8, 8))
        self.con2_batch_in = np.zeros((self.model_size[2], 8, 8))
        self.con1_batch_in = np.zeros((self.model_size[1], 24, 24))

    def cnn_forward(self, data):
        data = np.reshape(data, (self.input_size[0], self.input_size[1]))
        # convolution_1:
        for jj in range(self.model_size[0]):  # 1
            for kk in range(self.model_size[1]):  # 6
                alpha_1, B_1 = Binary_weight(self.weight_1[jj, kk, :, :])
                self.con1_out[kk, :, :] = alpha_1*convolve(data, B_1)  # 卷积
                # Batch_Norm
                self.con1_batch_out[kk, :, :] = Batch_Normalization(self.con1_out[kk, :, :], self.gamma_1[kk], self.beta_1[kk])
                # hard_tanh
                self.con1[kk, :, :] = hard_tanh(self.con1_batch_out[kk, :, :])  # sigmoid(Wx+b)
                # Batch_Normalization  conv--batch_norm--binary_tanh
                # pooling_1:
                self.pooling1_out[kk, :, :] = ave_pooling(self.con1[kk, :, :], [2, 2], 2)
        # convolution_2:   input:self.pooling1_out  [batch_size, 6, 12, 12]
        for kk in range(self.model_size[2]):  # 12
            t = np.zeros((8, 8))
            for jj in range(self.model_size[1]):  # 6
                alpha_2, B_2 = Binary_weight(self.weight_2[jj, kk, :, :])
                t += alpha_2*convolve(self.pooling1_out[jj, :, :], B_2)
            # ss = t/12
            self.con2_batch_in[kk, :, :] = t
            # Batch_Norm
            self.con2_batch_out[kk, :, :] = Batch_Normalization(self.con2_batch_in[kk, :, :], self.gamma_2[kk], self.beta_2[kk])
            self.con2_out[kk, :, :] = hard_tanh(self.con2_batch_out[kk, :, :])
            self.pooling2_out[kk, :, :] = ave_pooling(self.con2_out[kk, :, :], [2, 2], 2)
        # fully connected:
        self.full_con = np.reshape(self.pooling2_out, 192)  # 先一个4*4按行reshape为向量，在12个维度
        for kk in range(self.model_size[4]):
            # for jj in range(self.model_size[3]):
            # print(self.weight_out[:, kk].shape)
            alpha_out, B_out = Binary_weight(self.weight_out[:, kk][None, :])
            # print(B_out.shape)
            self.a[kk] = ((alpha_out*(self.full_con * B_out)).sum())
        self.t = Softmax(self.a)
        return self.t, data

    def BP(self, data, label):
        self.der_pool_1 = np.zeros((6, 12, 12))
        self.t, data = self.cnn_forward(data)
        # 梯度下降更新
        # for ii in range(self.batch_size):
        # Cost = -1*sum((math.log(self.t)*label))
        for ii in range(10):
            self.der_out[ii] = self.t[ii]*sum(label)-label[ii]
        # print(self.t, label, self.der_out)
        self.d_out_w = np.dot(self.full_con[:, None], self.der_out[None, :])
        self.der_full_con = self.full_con*(np.reshape(np.dot(self.weight_out, self.der_out[:, None]), 192))   # check here
        self.der_pool_2 = np.reshape(self.der_full_con, (12, 4, 4))  # 直接reshape就可以 顺序和正向刚好一样 哈哈
        for ii in range(self.model_size[2]):
            self.der_con_2[ii, :, :] = expand(self.der_pool_2[ii, :, :], 2)  # [12, 8, 8]
        self.der_con_2 = self.der_con_2*hard_tanh_derivative(self.con2_batch_out)   # self.con2_out*(1-self.con2_out)
        # batch norm der
        for ii in range(self.model_size[2]):
            self.der_gamma_2[ii], self.der_beta_2[ii], self.der_con_2[ii, :, :] = Batch_Normalization_derivative(self.con2_batch_in[ii, :, :], self.gamma_2[ii], self.beta_2[ii], self.der_con_2[ii, :, :])
        for ii in range(self.model_size[1]):
            for jj in range(self.model_size[2]):
                self.der_pool_1[ii, :, :] += deconvolution(self.der_con_2[jj, :, :], self.weight_2[ii, jj, :, :])
        for ii in range(self.model_size[1]):
            for jj in range(self.model_size[2]):
                self.d_con2_weight[ii, jj, :, :] = convolve(self.pooling1_out[ii, :, :], self.der_con_2[jj, :, :])
        for ii in range(self.model_size[1]):
            self.der_con_1[ii, :, :] = expand(self.der_pool_1[ii, :, :], 2)
        self.der_con_1 = self.der_con_1*hard_tanh_derivative(self.con1_batch_out)  #self.con1_out*(1-self.con1_out)
        for ii in range(self.model_size[1]):
            self.der_gamma_1[ii], self.der_beta_1[ii], self.der_con_1[ii, :, :] = Batch_Normalization_derivative(self.con1_out[ii, :, :], self.gamma_1[ii],
                                                                      self.beta_1[ii], self.der_con_1[ii, :, :])
        for ii in range(self.model_size[1]):
            self.d_con1_weight[0, ii, :, :] = convolve(data, self.der_con_1[ii, :, :])
            self.d_con1_bias[ii] = self.der_con_1[ii, :, :].sum()/(24*24)
        # update
        self.weight_1 -= self.learning_rate*self.d_con1_weight
        self.weight_2 -= self.learning_rate * self.d_con2_weight
        self.weight_out -= self.learning_rate * self.d_out_w
        self.gamma_1 -= self.learning_rate*self.der_gamma_1
        self.gamma_2 -= self.learning_rate * self.der_gamma_2
        self.beta_1 -= self.learning_rate*self.der_beta_1
        self.beta_2 -= self.learning_rate * self.der_beta_2


if __name__ == "__main__":
    model = cnn_model()
    # load data
    get_data = load_data.mnist(1)
    A = []
    for kk in range(model.iteration_number):
        print(kk)
        data, label = get_data.get_mini_bath("train")
        model.BP(data, np.reshape(label, 10))
        acc = 0
        if kk % 1000 == 0:
            for jj in range(1000):
                data, label = get_data.get_mini_bath("test")
                t, _ = model.cnn_forward(data)
                if np.argmax(t) == np.argmax(label):
                    acc += 1
                else:
                    continue
            print("Lenet Accuracy: ", acc / 1000)
            A.append(acc/1000)
    x = np.arange(len(A))
    plt.plot(x, A, "cs")
    plt.xlabel('iter')
    plt.ylabel('acc')
    plt.title('Binary-Network')
    plt.savefig("Binary-Network.jpg")
    plt.show()

#  acc 0.83




