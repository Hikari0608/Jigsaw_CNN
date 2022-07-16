from ctypes import util
from cv2 import repeat

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from testData import *
import time
from pathlib import Path
import cv2
from testData import *
from model import *
from scipy import ndimage
import os
import skimage
from Data import *

def loss_funtion(pred_im,gt):
    loss = []
    for pred_im_feature, gt_feature in zip(pred_im, gt):
        loss.append(tf.keras.losses.mae(pred_im_feature, gt_feature))

    return sum(loss)/len(loss)

def imshow(x, title=None, cbar=False, figsize=None):
    plt.figure(figsize=figsize)
    plt.imshow(np.squeeze(x), interpolation='nearest', cmap='gray')
    print(title)
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()

num_epochs = 5  #每个元素重复训练的次数
batch_size = 5  #每批次元素
learning_rate = 0.001   #学习速率

H = 480     #输入图片大小
W = 640
C = 3

ROOT_PATH = "F:\\UIEB_end\\resize"      #数据目录
train_data_directory = os.path.join(ROOT_PATH, "train")
test_data_directory = os.path.join(ROOT_PATH, "test")

rep = False     #重复训练

model = DnCNN()
if(rep == True):
    model.build([batch_size,H,W,C])
    model.load_weights(r'model.h5') 
else:
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

images = load_data(train_data_directory)
#optimier = tf.keras.optimizers.Adam(learning_rate=learning_rate)

for i in range(num_epochs):
    for batch in range(0,len(images)//batch_size):
        #np.random.seed(seed=0)  # for reproducibility
        #X += np.random.normal(0, 0.01, X.shape) # add AWGN)
        #imshow(X)
        input,gt = load_img(train_data_directory,images,batch,batch_size)
        #input = tf.constant()
        x = tf.concat(input,axis=0)
        y = tf.concat(gt,axis=0)

        #for x,y in zip(input,gt):
        with tf.GradientTape() as tape:
            pred = model(x)
            loss = tf.reduce_sum(loss_funtion(pred,y))
            print("loss %f"%(loss.numpy()))
            psnr = tf.reduce_sum(tf.image.psnr(pred,y,1.0))/batch_size
            ssim = tf.reduce_sum(tf.image.ssim(pred,y,1.0))/batch_size
            print("ssim: %f ,psnr: %f"%(ssim.numpy(),psnr.numpy()))
            #imshow(gt)
        grads = tape.gradient(loss,model.variables)
        #optimier.apply_gradients(grads_and_vars=zip(grads,model.variables))
        model.optimizer.apply_gradients(grads_and_vars=zip(grads,model.variables))

model.save_weights('model.h5')
