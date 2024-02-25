import pandas as pd
import yaml, os
from pathlib import Path
import keras
import tensorflow as tf
import numpy as np
import librosa, cv2
from wav_to_spectrogram_utils import wav_to_stft, get_windows

class DataLoader():
    'Generates data for Keras'
    def __init__(self, path_to_slakh, instr_class, batch_size=32, nbr_folds = 5, shuffle=False, augment=False) -> None:
        self.path_to_slakh = Path(path_to_slakh)
        self.instr_class = instr_class
        self.batch_size = batch_size
        self.nbr_folds = nbr_folds
        self.shuffle = shuffle
        self.augment = augment
        self.metadata_df = self.get_metadata()
        self.data = self.get_data()

    def get_metadata(self):
        metadata_df = pd.DataFrame(columns = ['mix_path'])
        for i,path in enumerate(self.path_to_slakh.glob('**/metadata.yaml')):
            root = path.parent
            mix_path = root / 'mix.wav'
            metadata_df.loc[i, 'mix_path'] = mix_path
            metadata_path = root / 'metadata.yaml'
            with open(metadata_path, 'r') as file:
                metadata = yaml.safe_load(file)
            j=0
            for key in metadata['stems'].keys():
                if metadata['stems'][key]['inst_class'] == self.instr_class:
                    metadata_df.loc[i, f'instr_path_{j}'] = root / f'stems/{key}.wav'
                    j+=1

        return metadata_df

    def get_data(self):
        if len([i for i in self.path_to_slakh.glob(f'**/{self.instr_class}_windows')])==0 or len([i for i in self.path_to_slakh.glob(f'**/{self.instr_class}_windows')])!=len([i for i in self.path_to_slakh.glob('**/mix_windows')]):
            print(1)
            for index, row in self.get_metadata.iterows():
                mix_path, instr_path = row['mix_path'], row['instr_path']
                Intensity_Stft, Angle_Stft, n, sr = wav_to_stft(instr_path)
                Intensity_Stft = librosa.amplitude_to_db(Intensity_Stft, ref=0)
                Intensity_Stft = Intensity_Stft.astype(np.uint8)
                windows = get_windows(Intensity_Stft, 64, 128)
                os.makedirs(mix_path.parent / f'Spectrograms/{self.instr_class}_windows/', exist_ok=True)
                for k, window in enumerate(windows):
                    if os.path.isfile(mix_path.parent / f'Spectrograms/{self.instr_class}_windows/{k}.jpg'):
                        os.remove(mix_path.parent / f'Spectrograms/{self.instr_class}_windows/{k}.jpg')
                    cv2.imwrite(mix_path.parent / f'Spectrograms/{self.instr_class}_windows/{k}.jpg', window)

                Intensity_Stft, Angle_Stft, n, sr = wav_to_stft(mix_path)
                Intensity_Stft = librosa.amplitude_to_db(Intensity_Stft, ref=0)
                windows = get_windows(Intensity_Stft, 64, 128)
                os.makedirs(mix_path.parent / f'Spectrograms/mix_windows/', exist_ok=True)
                for k, window in enumerate(windows):
                    if os.path.isfile(mix_path.parent / f'Spectrograms/mix_windows/{k}.jpg'):
                        os.remove(mix_path.parent / f'Spectrograms/mix_windows/{k}.jpg')
                    cv2.imwrite(mix_path.parent / f'Spectrograms/mix_windows/{k}.jpg', window)


        for i in range(len(self.metadata_df)):
            mix_dir = self.metadata_df.iloc[i]['mix_path'].parent / 'Spectrograms' / f'mix_windows'
            instr_dir = self.metadata_df.iloc[i]['mix_path'].parent / 'Spectrograms' / f'{self.instr_class}_windows'
            if i == 0:
                X = keras.preprocessing.image_dataset_from_directory(mix_dir,
                                                                    labels=None,
                                                                    color_mode= "grayscale",
                                                                    image_size=(512, 128), shuffle = False)
                y = keras.preprocessing.image_dataset_from_directory(instr_dir,
                                                                    labels=None,
                                                                    color_mode= "grayscale",
                                                                    image_size=(512, 128), shuffle = False)
            else:
                X = X.concatenate(keras.preprocessing.image_dataset_from_directory(mix_dir,
                                                                                    labels=None,
                                                                                    color_mode= "grayscale",
                                                                                    image_size=(512, 128), shuffle = False))
                y = y.concatenate(keras.preprocessing.image_dataset_from_directory(instr_dir,
                                                                                    labels=None,
                                                                                    color_mode= "grayscale",
                                                                                    image_size=(512, 128), shuffle = False))

        if self.shuffle:
            return (tf.data.Dataset.zip((X, y)).shuffle(buffer_size=1000)).batch(self.batch_size)
        else:
            return tf.data.Dataset.zip((X, y)).batch(self.batch_size)

    def cv_split_dataset(self, fold):
        fold_size = len(self.data) // self.num_folds
        start_index = fold * fold_size
        end_index = (fold + 1) * fold_size

        train_data = (self.data).take(start_index).concatenate((self.data).skip(end_index))
        val_data = (self.data).skip(start_index).take(end_index - start_index)

        return train_data, val_data





if __name__=='__main__':
    data = DataLoader('raw_data/babyslakh_16k', 'Drums')
    print(data.metadata_df)
