import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np

def downsample_block(x, n_filters):
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 5,
                     strides=2,
                     padding='same',
                     kernel_initializer = "he_normal")(x)
   x = layers.BatchNormalization(axis=-1,
                                 momentum=0.01,
                                epsilon=1e-3)(x)
   x = layers.LeakyReLU(alpha=0.2)(x)
   return x


def upsample_block(x, conv_features, n_filters, dropout, activation_func):

    # upsample
    x = layers.Conv2DTranspose(n_filters, 5, strides=2, padding='same', kernel_initializer = "he_normal")(x)
    if activation_func == 'relu':
        x = layers.ReLU()(x)
    elif activation_func == 'sigmoid':
        x = keras.activations.sigmoid(x)
    else:
        pass
    x = layers.BatchNormalization(axis=-1)(x)
    # dropout
    if dropout:
        x = layers.Dropout(0.5)(x)
    # concatenate
    if conv_features != None:
        x = layers.Concatenate(axis=-1)([x, conv_features])
    return x

def build_unet_model(mean, stddev):
    # inputs
    inputs = layers.Input(shape=(256,128,1))
    #normalization layer
    norm_inputs = layers.Normalization(mean=mean, variance=stddev**2)(inputs)
    # encoder: contracting path - downsample
    p1 = downsample_block(norm_inputs, 16)
    # 2 - downsample
    p2 = downsample_block(p1, 32)
    # 3 - downsample
    p3 = downsample_block(p2, 64)
    # 4 - downsample
    p4 = downsample_block(p3, 128)
    p5 = downsample_block(p4, 256)
    # 5 - bottleneck
    bottleneck = downsample_block(p5, 512)
    # decoder: expanding path - upsample
    # 1 - upsample
    u1 = upsample_block(bottleneck, p5, 256, True, 'relu')
    # 2 - upsample
    u2 = upsample_block(u1, p4, 128, True, 'relu')
    # 3 - upsample
    u3 = upsample_block(u2, p3, 64, False, 'relu')
    # 4 - upsample
    u4 = upsample_block(u3, p2, 32, False, 'relu')
    # 5 - upsample
    u5 = upsample_block(u4, p1, 16, False, 'relu')
    # 6 - upsample
    u6 = upsample_block(u5, None, 1, False, 'relu')
    u7 = layers.Conv2D(
            1,
            (4, 4),
            dilation_rate=(2, 2),
            activation="sigmoid",
            padding="same",
            kernel_initializer="he_normal",
        )(u6)
    # outputs
    outputs = layers.Multiply()([u7, inputs])
    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
    unet_model.compile(optimizer="adam",
                        loss=tf.keras.metrics.mean_absolute_error)

    return unet_model
