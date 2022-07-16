from random import sample
import tensorflow as tf

class DnCNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.bn1 = tf.keras.layers.BatchNormalization(axis=0)
        self.bn2 = tf.keras.layers.BatchNormalization(axis=0)
        self.bn3 = tf.keras.layers.BatchNormalization(axis=0)
        self.bn4 = tf.keras.layers.BatchNormalization(axis=0)
        self.bn5 = tf.keras.layers.BatchNormalization(axis=0)
        self.bn6 = tf.keras.layers.BatchNormalization(axis=0)
        self.relu = tf.keras.layers.ReLU()
        self.dconv1 = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=3,
            dilation_rate=1,
            padding='same'
            
        )   
        self.dconv2 = tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=3,
            dilation_rate=2,
            padding='same'
        )
        self.dconv3 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=3,
            dilation_rate=3,
            padding='same'
        )
        self.dconv4 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            dilation_rate=4,
            padding='same'
        )
        self.dconv5 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=3,
            dilation_rate=3,
            padding='same'
        )
        self.dconv6 = tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=3,
            dilation_rate=2,
            padding='same'
        )
        self.dconv7 = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=3,
            dilation_rate=1,
            padding='same'
        )

    def call(self,x):
        x = self.dconv1(x)
        x = self.relu(x)
        x = self.dconv2(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dconv3(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dconv4(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dconv5(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dconv6(x)
        x = self.bn5(x)
        x = self.relu(x)
        ret = self.dconv7(x)
        return ret