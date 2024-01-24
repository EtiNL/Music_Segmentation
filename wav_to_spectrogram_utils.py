import librosa
import soundfile as sf
import numpy as np

def wav_to_stft(track_path):
    y, sr = librosa.load(track_path)
    n = len(y)
    n_fft = 1022
    y_pad = librosa.util.fix_length(y, size=n + n_fft // 2)
    Stft = librosa.stft(y_pad, n_fft=n_fft)
    Angle_Stft = np.angle(Stft)
    Intensity_Stft = np.abs(Stft)
    return Intensity_Stft, Angle_Stft, n, sr


def sft_to_wav(path, Intensity_Stft, Angle_Stft, n, sr):
    Phase_Stft = np.exp(1j *Angle_Stft)
    Stft = np.multiply(Intensity_Stft,Phase_Stft)
    y_out = librosa.istft(Stft, length=n)
    sf.write(path, y_out, sr, subtype='PCM_24')


def get_windows(Stft, step_size, window_size):
    # get the window and image sizes
    image_h, image_w = Stft.shape
    Sliding_windows = []

    # loop over the image, taking steps of size `step_size`
    for x in range(0, image_w, step_size):
        # define the window
        if x+window_size < image_w:
            window = Stft[:, x:x + window_size]
            Sliding_windows.append(window)
    return Sliding_windows

def get_Stft_from_windows(Windows, step_size, window_size):
    """_summary_

    Args:
        Windows (_type_): _description_
        step_size (_type_): _description_
        window_size (_type_): _description_

    Returns:
        _type_: _description_
    """
    y = []
    for i,window in enumerate(Windows):
        for j,timestep_Stft in enumerate(window):
            id = i*step_size + j
            if len(y) <= id:
                y.append([timestep_Stft])
            else:
                y[id].append(timestep_Stft)

    Stft = []
    for i in len(y):
        Stft.append(np.mean(y[i]))

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
