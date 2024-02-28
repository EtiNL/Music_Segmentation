from main import wav_to_spectrogram_utils as ws

# def test_wav_to_stft_to_wav():

def test_get_Stft_from_windows():
    Intensity_Stft, Angle_Stft, n, sr = ws.wav_to_stft('raw_data/test_data/Dani_California.wav')
    windows, pad_size = ws.get_windows(Intensity_Stft, 64, 128)
    print(pad_size)
    Intensity_Stft_res = ws.get_Stft_from_windows(windows, 64, pad_size)
    assert Intensity_Stft.shape == Intensity_Stft_res.shape
    assert (Intensity_Stft-Intensity_Stft_res).all()==0
