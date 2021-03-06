from virtual_data import get_train_data
import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Lambda, Layer, BatchNormalization, Activation,concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.layers import LeakyReLU
import os
import random
import time
import wget
import tarfile
import numpy as np
import cv2
from PIL import Image
import os 
import sys
import glob
import json
import matplotlib.pyplot as plt 
from labelme import utils
import imgviz

class RTNet:
    def __init__(self, train_stage=1,use_cpu=False, print_summary=False):
        self.train_stage = 1
        self.parameter = [24,48,64,96,128,196]
        self.build(use_cpu=use_cpu, print_summary=print_summary)


    def predict(self, image):
        return self.model.predict(np.array([image]))
    
    def save(self, file_path='model.h5'):
        self.model.save_weights(file_path)
        
    def load(self, file_path='model.h5'):
        self.model.load_weights(file_path)
    
    def BatchGenerator(self,batch_size=8, image_size=(640, 640, 5), labels=1):#500, 375
        while True:
            images,truths = get_train_data(image_size[2])
            yield images, truths
            
    def train(self, epochs=10, steps_per_epoch=50,batch_size=32):
        batch_generator = self.BatchGenerator(batch_size=batch_size)
        self.model.fit_generator(batch_generator, steps_per_epoch=steps_per_epoch, epochs=epochs)

    def build_conv2D_block(self, inputs, filters, kernel_size, strides, block, i):
        conv2d = Conv2D(filters = filters, kernel_size=kernel_size,strides=strides, padding='same', \
        name='conv{}-{}'.format(block, i), use_bias=True,bias_initializer='zeros')(inputs)
        conv2d = BatchNormalization(name='batchnorm{}-{}'.format(block, i))(conv2d)
        conv2d_output = Activation(LeakyReLU(alpha=0.1), name='relu{}-{}'.format(block, i))(conv2d)
        return conv2d_output

    def build_conv2Dtranspose_block(self, inputs, filters, kernel_size, strides, block, i):
        conv2d = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, use_bias=True,bias_initializer='zeros', padding='same', \
            name='deconv4{}-{}'.format(block, i))(inputs)
        conv2d = BatchNormalization(name='batchnorm_decon{}-{}'.format(block, i))(conv2d)
        conv2d_deconv = Activation(LeakyReLU(alpha=0.1), name='relu_decon{}-{}'.format(block, i))(conv2d)
        return conv2d_deconv

    def my_loss_error(self, y_true, y_pred):
        return K.sum((K.abs(y_pred - y_true)))

    def build(self, use_cpu=False, print_summary=False):
        inputs = Input(shape=(640, 640, 5))
                        
        # initial layer
        conv2d_conv0_1 = self.build_conv2D_block(inputs,        filters = self.parameter[0],kernel_size=1,strides=1, block=0, i=0)
        conv2d_conv0   = self.build_conv2D_block(conv2d_conv0_1,filters = self.parameter[0],kernel_size=3,strides=1, block=0, i=1)
        ###########
        conv2d_conv0   = self.build_conv2D_block(conv2d_conv0,filters = self.parameter[0],kernel_size=3,strides=1, block=0, i=2)
        conv2d_conv0   = self.build_conv2D_block(conv2d_conv0,filters = self.parameter[0],kernel_size=3,strides=1, block=0, i=3)
        conv2d_conv0   = self.build_conv2D_block(conv2d_conv0,filters = self.parameter[0],kernel_size=3,strides=1, block=0, i=4)
        # first conv layer
        conv2d_conv1_1 = self.build_conv2D_block(conv2d_conv0,  filters = self.parameter[1],kernel_size=3,strides=2, block=1, i=0)
        conv2d_conv1   = self.build_conv2D_block(conv2d_conv1_1,filters = self.parameter[1],kernel_size=3,strides=1, block=1, i=1)
        ###########
        conv2d_conv1   = self.build_conv2D_block(conv2d_conv1,filters = self.parameter[1],kernel_size=3,strides=1, block=1, i=2)
        conv2d_conv1   = self.build_conv2D_block(conv2d_conv1,filters = self.parameter[1],kernel_size=3,strides=1, block=1, i=3)
        conv2d_conv1   = self.build_conv2D_block(conv2d_conv1,filters = self.parameter[1],kernel_size=3,strides=1, block=1, i=4)
        # second conv layer
        conv2d_conv2_2 = self.build_conv2D_block(conv2d_conv1,  filters = self.parameter[2],kernel_size=3,strides=2, block=2, i=0)
        conv2d_conv2_1 = self.build_conv2D_block(conv2d_conv2_2,filters = self.parameter[2],kernel_size=3,strides=1, block=2, i=1)
        conv2d_conv2   = self.build_conv2D_block(conv2d_conv2_1,filters = self.parameter[2],kernel_size=3,strides=1, block=2, i=2)
        ###########
        conv2d_conv2   = self.build_conv2D_block(conv2d_conv2,filters = self.parameter[2],kernel_size=3,strides=1, block=2, i=3)
        conv2d_conv2   = self.build_conv2D_block(conv2d_conv2,filters = self.parameter[2],kernel_size=3,strides=1, block=2, i=4)
        conv2d_conv2   = self.build_conv2D_block(conv2d_conv2,filters = self.parameter[2],kernel_size=3,strides=1, block=2, i=5)
        # third conv layer
        conv2d_conv3_2 = self.build_conv2D_block(conv2d_conv2,  filters = self.parameter[3],kernel_size=3,strides=2, block=3, i=0)
        conv2d_conv3_1 = self.build_conv2D_block(conv2d_conv3_2,filters = self.parameter[3],kernel_size=3,strides=1, block=3, i=1)
        conv2d_conv3   = self.build_conv2D_block(conv2d_conv3_1,filters = self.parameter[3],kernel_size=3,strides=1, block=3, i=2)
        ###########
        conv2d_conv3   = self.build_conv2D_block(conv2d_conv3,filters = self.parameter[3],kernel_size=3,strides=1, block=3, i=3)
        conv2d_conv3   = self.build_conv2D_block(conv2d_conv3,filters = self.parameter[3],kernel_size=3,strides=1, block=3, i=4)
        conv2d_conv3   = self.build_conv2D_block(conv2d_conv3,filters = self.parameter[3],kernel_size=3,strides=1, block=3, i=5)
        # fourth conv layer
        conv2d_conv4_2 = self.build_conv2D_block(conv2d_conv3,  filters = self.parameter[4],kernel_size=3,strides=2, block=4, i=0)
        conv2d_conv4_1 = self.build_conv2D_block(conv2d_conv4_2,filters = self.parameter[4],kernel_size=3,strides=1, block=4, i=1)
        conv2d_conv4   = self.build_conv2D_block(conv2d_conv4_1,filters = self.parameter[4],kernel_size=3,strides=1, block=4, i=2)
        ###########
        conv2d_conv4   = self.build_conv2D_block(conv2d_conv4,filters = self.parameter[4],kernel_size=3,strides=1, block=4, i=3)
        conv2d_conv4   = self.build_conv2D_block(conv2d_conv4,filters = self.parameter[4],kernel_size=3,strides=1, block=4, i=4)
        conv2d_conv4   = self.build_conv2D_block(conv2d_conv4,filters = self.parameter[4],kernel_size=3,strides=1, block=4, i=5)
        # fifth conv layer
        conv2d_conv5_1 = self.build_conv2D_block(conv2d_conv4,  filters = self.parameter[5],kernel_size=3,strides=2, block=5, i=0)
        conv2d_conv5   = self.build_conv2D_block(conv2d_conv5_1,filters = self.parameter[5],kernel_size=3,strides=1, block=5, i=1)
        ###########
        conv2d_conv5   = self.build_conv2D_block(conv2d_conv5,filters = self.parameter[5],kernel_size=3,strides=1, block=5, i=2)
        conv2d_conv5   = self.build_conv2D_block(conv2d_conv5,filters = self.parameter[5],kernel_size=3,strides=1, block=5, i=3)
        conv2d_conv5   = self.build_conv2D_block(conv2d_conv5,filters = self.parameter[5],kernel_size=3,strides=1, block=5, i=4)
        # fifth deconv layer
        conv2d_deconv5_1 = self.build_conv2D_block(conv2d_conv5,filters = self.parameter[5],kernel_size=3,strides=1, block=55, i=0)
        conv2d_deconv4   = self.build_conv2Dtranspose_block(conv2d_deconv5_1, filters=self.parameter[4], kernel_size=4, strides=2, block=55, i=1)
            
        #Concat4
        Concat_concat4 = concatenate([conv2d_conv4, conv2d_deconv4] , axis=-1)
            
        # fourth deconv layer
        conv2d_deconv4_1 = self.build_conv2D_block(Concat_concat4,filters = self.parameter[4],kernel_size=3,strides=1, block=44, i=0)
        conv2d_deconv3   = self.build_conv2Dtranspose_block(conv2d_deconv4_1, filters=self.parameter[3], kernel_size=4, strides=2, block=44, i=1)
            
        #Concat3
        Concat_concat3 = concatenate([conv2d_conv3 , conv2d_deconv3] , axis=-1)
            
        # third deconv layer
        conv2d_deconv3_1 = self.build_conv2D_block(Concat_concat3,filters = self.parameter[3],kernel_size=3,strides=1, block=33, i=0)
        conv2d_deconv2   = self.build_conv2Dtranspose_block(conv2d_deconv3_1, filters=self.parameter[2], kernel_size=4, strides=2, block=33, i=1)
            
        #Concat2
        Concat_concat2 = concatenate([conv2d_conv2 , conv2d_deconv2] , axis=-1)
            
        # sencod deconv layer
        conv2d_deconv2_1 = self.build_conv2D_block(Concat_concat2,filters = self.parameter[2],kernel_size=3,strides=1, block=22, i=0)
        conv2d_deconv1   = self.build_conv2Dtranspose_block(conv2d_deconv2_1, filters=self.parameter[1], kernel_size=4, strides=2, block=22, i=1)
            
        #Concat1
        Concat_concat1 = concatenate([conv2d_conv1 , conv2d_deconv1] , axis=-1)
            
        # first deconv layer
        conv2d_deconv1_1 = self.build_conv2D_block(Concat_concat1,filters = self.parameter[1],kernel_size=3,strides=1, block=11, i=0)
        conv2d_deconv0   = self.build_conv2Dtranspose_block(conv2d_deconv1_1, filters=self.parameter[0], kernel_size=4, strides=2, block=11, i=1)


        output = Conv2DTranspose(filters=1, kernel_size=1, strides=1, activation='sigmoid', padding='same', name='output')(conv2d_deconv0)
        print(output.shape)
        self.model = Model(inputs=inputs, outputs=output)
        # self.model.summary()
        self.model.compile(optimizer='adam',loss=self.my_loss_error,metrics=['accuracy', 'mse'])
