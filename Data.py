from random import shuffle
import os
import skimage
import numpy as np

dataRandow = True
showData = False

def load_data(data_directory):
    directories = [d for d in os.listdir(os.path.join(data_directory,"gt"))
                   if d.endswith(".png")]
    
    if(dataRandow == True):
        shuffle(directories)

    return directories

def load_img(data_directory,directories,cnt,batch_size):
    input_img = []
    gt_img = []

    L = batch_size*cnt
    for d in directories[L:L+batch_size:]:
        input_img.append(np.expand_dims(skimage.io.imread(os.path.join(data_directory,"input",d)).astype(np.float32)/255.0,axis=0))
        gt_img.append(np.expand_dims(skimage.io.imread(os.path.join(data_directory,"gt",d)).astype(np.float32)/255.0,axis=0))
        #input_img.append(np.expand_dims(skimage.io.imread(os.path.join(data_directory,"input",d)).astype(np.float32)/255.0,axis=3))
        #gt_img.append(np.expand_dims(skimage.io.imread(os.path.join(data_directory,"gt",d)).astype(np.float32)/255.0,axis=3))

    return input_img,gt_img
