import tensorflow as tf
import numpy as np
import librosa, cv2
import matplotlib.pyplot as plt
from wav_to_spectrogram_utils import *
from Unet_Model import build_unet_model
from dataLoader import configure_for_performance
import os, keras
import soundfile as sf

def denormalize(normalized_features, mean, stddev):
    denormalized_features = normalized_features * stddev + mean
    return denormalized_features

def get_mean_and_std_of_normalization_layer(model):
    print('OK')
    for layer in model.layers:
        print(1)
        if isinstance(layer, keras.layers.Normalization):
            mean = layer.mean.numpy()
            stddev = np.sqrt(layer.variance.numpy())
            return mean, stddev


def main(wav_path, model_weights_paths):

    Intensity_Stft, Angle_Stft, n, sr, mean, stddev = wav_to_stft(wav_path)
    spec = 2*np.log10(1+Intensity_Stft)
    windows, pad_size = get_windows(spec, 64, 128)

    X = tf.data.Dataset.from_tensor_slices(windows)

    test_X = configure_for_performance(X, 8, 'test')

    for i, model_path in enumerate(model_weights_paths):
        model = build_unet_model(0.6176444333988994,0.8785275373835086)
        model.load_weights(model_path)

        y_test = model.predict(test_X)

        windows_res = [ np.reshape(denormalize(y_test[k], 0.6176444333988994, 0.8785275373835086),(256,128)) for k in range(y_test.shape[0])]
        spec_res = get_Stft_from_windows(windows_res, 64, pad_size)
        Intensity_Stft_res = np.power(10,spec_res/2) - 1
        signal = denormalize(sft_to_signal(Intensity_Stft_res, Angle_Stft, n, sr), mean, stddev)
        sf.write(f'raw_data/test_data/Dani_California_drums_{i}.wav', signal, sr, subtype='PCM_24')




if __name__=='__main__':
    main('raw_data/test_data/Dani_California.wav', ['Model_weights/Unet_Drums_f0.h5'])
