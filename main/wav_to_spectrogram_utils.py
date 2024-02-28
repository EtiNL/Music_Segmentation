import librosa
import numpy as np
import soundfile as sf

def wav_to_stft(track_path):
    y, sr = librosa.load(track_path)
    mean, stddev = np.mean(y), np.std(y)
    y = (y-mean)/stddev
    n = len(y)
    n_fft = 510
    y_pad = librosa.util.fix_length(y, size=n + n_fft // 2)
    Stft = librosa.stft(y_pad, n_fft=n_fft)
    Angle_Stft = np.angle(Stft)
    Intensity_Stft = np.abs(Stft)
    return Intensity_Stft, Angle_Stft, n, sr, mean, stddev


def sft_to_wav(path, Intensity_Stft, Angle_Stft, n, sr):
    Phase_Stft = np.exp(1j *Angle_Stft)
    Stft = np.multiply(Intensity_Stft,Phase_Stft)
    y_out = librosa.istft(Stft, length=n)
    sf.write(path, y_out, sr, subtype='PCM_24')

def sft_to_signal(Intensity_Stft, Angle_Stft, n, sr):
    Phase_Stft = np.exp(1j *Angle_Stft)
    Stft = np.multiply(Intensity_Stft,Phase_Stft)
    y_out = librosa.istft(Stft, length=n)
    # sf.write(path, y_out, sr, subtype='PCM_24')
    return y_out


def get_windows(Stft, step_size, window_size):
    # get the window and image sizes
    image_h, image_w = Stft.shape
    Sliding_windows = []

    if image_w % step_size !=0:
        Stft = np.concatenate((Stft,np.zeros((image_h, step_size - image_w % step_size))), axis = 1)
        image_w_temp = image_w + step_size - image_w % step_size

    # loop over the image, taking steps of size `step_size`
    for x in range(0, image_w_temp, step_size):
        # define the window
        if x+window_size <= image_w_temp:
            window = Stft[:, x:x + window_size].reshape((256,128,1))
            Sliding_windows.append(window)
    return Sliding_windows, step_size - image_w % step_size

def get_mean(L):
    L = np.array(L)
    return np.mean(L, axis = 0)

def get_Stft_from_windows(Windows, step_size, pad_size):
    y = []
    for i,window in enumerate(Windows):
        for j in range(window.shape[1]):
            timestep_Stft = window[:,j]
            id = i*step_size + j
            if len(y) <= id:
                y.append([timestep_Stft])
            else:
                (y[id]).append(timestep_Stft)

    Stft = []
    for i in range(len(y)):
        Stft.append(get_mean(y[i]))
    Stft = np.transpose(np.array(Stft))[:,:-pad_size]
    return Stft

if __name__ == '__main__':
    for i in range(20):
        if i < 9:
            path = f'babyslakh_16k/babyslakh_16k/Track0000{i+1}/mix.wav'
        else:
            path = f'babyslakh_16k/babyslakh_16k/Track000{i+1}/mix.wav'
        Intensity_Stft, Angle_Stft, n, sr = wav_to_stft(path)
        windows = get_windows(Intensity_Stft, 64, 128)

        print(len(windows),windows[0].shape, n, sr)
