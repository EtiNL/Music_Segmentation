import tensorflow as tf
import keras
from keras import layers
# import matplotlib.pyplot as plt
# import numpy as np

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

def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(waveform, frame_length=512, frame_step=128, fft_length = 510, pad_end= True, window_fn=tf.signal.hamming_window)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return tf.math.log1p(tf.abs(spectrogram)), tf.math.angle(spectrogram)

def build_unet_stft_model():
    # inputs
    inputs = layers.Input(shape=(2**15)) # ceil(log2(22050)) = 15 is the sampling rate of my training set and also half the sampling rate of standard CD quality

    mag_spec, angle_spec = get_spectrogram(inputs)

    # # #Normalize spectrograms
    # bn_layer = layers.BatchNormalization(axis=-1)
    # x = bn_layer(mag_spec)
    # # Retrieve the mean and standard deviation learned by the BatchNormalization layer
    # mean = bn_layer.get_weights()[0]
    # std = bn_layer.get_weights()[1]


    # encoder: contracting path - downsample
    p1 = downsample_block(mag_spec, 16)
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
            kernel_initializer="he_normal")(u6)

    # outputs (signal_reconstruction)
    outputs_mag_spec = tf.math.expm1(layers.Multiply()([u7, mag_spec]))
    outputs_spec = tf.math.multiply(tf.cast(outputs_mag_spec, tf.complex64), tf.complex(tf.cos(angle_spec),tf.sin(angle_spec)))
    outputs = tf.signal.inverse_stft(tf.squeeze(outputs_spec, axis=-1), frame_length=512, frame_step=128, fft_length = 510,
                                     window_fn=tf.signal.inverse_stft_window_fn(128, forward_window_fn=tf.signal.hamming_window))[:,:inputs.shape[1]]


    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
    unet_model.compile(optimizer="adam",
                        loss=tf.keras.metrics.mean_absolute_error)

    return unet_model


if __name__=='__main__':
    build_unet_stft_model()
    # frame_length = 400
    # frame_step = 160
    # window_fn = tf.signal.hamming_window
    # waveform = tf.random.normal(dtype=tf.float32, shape=[1000])
    # stft = tf.signal.stft(
    #     waveform, frame_length, frame_step, window_fn=window_fn)
    # inverse_stft = tf.signal.inverse_stft(
    #     stft, frame_length, frame_step,
    #     window_fn=tf.signal.inverse_stft_window_fn(
    #     frame_step, forward_window_fn=window_fn))
    # print(inverse_stft.shape)
