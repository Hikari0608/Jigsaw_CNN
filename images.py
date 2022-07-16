import tensorflow as tf
from Data import *
from model import *
from matplotlib import pyplot as plt

def imread(pth,name):
    ret = tf.io.read_file(os.path.join(pth,name))
    ret = tf.io.encode_png(ret)
    return ret

def imsave(img,name,pth):
    img = tf.io.encode_png(img)
    tf.io.write_file(os.path.join(pth,name),img)

def imshow(x, title=None, cbar=False, figsize=None):
    plt.figure(figsize=figsize)
    plt.imshow(np.squeeze(x), interpolation='nearest', cmap='gray')
    print(title)
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()

ROOT_PATH = "F:\\UIEB_end\\resize\\train"
gt_data_directory = os.path.join(ROOT_PATH, "gt")
input_data_directory = os.path.join(ROOT_PATH, "input")

images = load_data(ROOT_PATH)
batch_size = 5
H = 480
W = 640
C = 3

model = DnCNN()
model.build([batch_size,H,W,C])
model.load_weights(r'model.h5') 


for batch in range(0,len(images)//batch_size):
        #np.random.seed(seed=0)  # for reproducibility
        #X += np.random.normal(0, 0.01, X.shape) # add AWGN)
        #imshow(X)
        input,gt = load_img(ROOT_PATH,images,batch,batch_size)
        #input = tf.constant()
        x = tf.concat(input,axis=0)
        y = tf.concat(gt,axis=0)

        #for x,y in zip(input,gt):
        with tf.GradientTape() as tape:
            pred = model(x)
            x_img = tf.split(x,batch_size,axis=0)
            pred_img = tf.split(pred,batch_size,axis=0)
            y_img = tf.split(y,batch_size,axis=0)

            for i,j,k in zip(x_img,pred_img,y_img):
                i = tf.squeeze(i)
                j = tf.squeeze(j)
                k = tf.squeeze(k)
                imshow(tf.concat([i,j,k],axis=1),title="IMG / PRED IMG / GT")
