import tensorflow as tf
import numpy as np
import librosa
from Unet_Model import *
from dataLoader import configure_for_performance
import soundfile as sf

class predict_DataLoader():
    def __init__(self, path_to_file, batch_size = 8) -> None:
        self.path_to_file = path_to_file
        self.batch_size = batch_size

    def get_data(self):

        y, sr = librosa.load(self.path_to_file)
        self.len_signal = len(y)

        if sr != 22050:
          y = librosa.resample( y, orig_sr = sr, target_sr = 22050)

        mean, stddev = np.mean(y), np.std(y)
        y = (y-mean)/stddev


        X = tf.signal.frame(y,
                            2**15,
                            2**13,
                            pad_end=True,
                            pad_value=0,
                            axis=-1)

        return configure_for_performance(tf.data.Dataset.from_tensor_slices(X), self.batch_size, 'test'), mean, stddev


def get_signal_from_frames(Frames, step_size, input_signal_len):
    y = []
    for i in range(Frames.shape[0]):
        frame = Frames[i,:]
        for j in range(frame.shape[0]):
            timestep_frame = frame[j]
            id = i*step_size + j
            if len(y) <= id:
                y.append([timestep_frame])
            else:
                (y[id]).append(timestep_frame)

    signal = []
    for i in range(len(y)):
        signal.append(np.mean(y[i]))
    signal = np.array(signal)[:input_signal_len]
    return signal



def predict(wav_path, model_weights_paths):

    data_loader = predict_DataLoader(wav_path)
    X, mean, stddev = data_loader.get_data()
    print(mean, stddev)
    len_input_signal = data_loader.len_signal

    signals = []
    for i, model_path in enumerate(model_weights_paths):
        model = build_unet_stft_model()
        model.load_weights(model_path)

        y = model.predict(X)
        signals.append(get_signal_from_frames(y, 2**13, len_input_signal))

    signal = np.mean(np.array(signals), axis = 0)*stddev + mean
    return signal
