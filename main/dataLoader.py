import pandas as pd
import yaml
from pathlib import Path
import tensorflow as tf
import numpy as np
from wav_to_spectrogram_utils import wav_to_stft, get_windows

def configure_for_performance(ds, batch_size, mode):
  ds = ds.cache()
  if mode == 'train':
    ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
  return ds

class DataLoader():
    'Generates data for Keras'
    def __init__(self, path_to_slakh, instr_class, batch_size = 8) -> None:
        self.path_to_slakh = Path(path_to_slakh)
        self.instr_class = instr_class
        self.batch_size = batch_size
        self.metadata_df = self.get_metadata()

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

    def get_data(self, indexes, mode):
        X_slices = []
        y_slices = []

        for index, row in self.metadata_df.iloc[indexes].iterrows():
            mix_path, instr_path = row['mix_path'], row['instr_path_0']
            Intensity_Stft, Angle_Stft, n, sr, mean, stddev = wav_to_stft(instr_path)
            spec = 2*np.log10(1+Intensity_Stft)
            windows, pad_size = get_windows(spec, 64, 128)
            y_slices += windows


            Intensity_Stft, Angle_Stft, n, sr, mean, stddev = wav_to_stft(mix_path)
            spec = 2*np.log10(1+Intensity_Stft)
            windows, pad_size = get_windows(spec, 64, 128)
            X_slices += windows


        X = tf.data.Dataset.from_tensor_slices(X_slices)
        y = tf.data.Dataset.from_tensor_slices(y_slices)

        return configure_for_performance(tf.data.Dataset.zip((X, y)), self.batch_size, mode)




if __name__=='__main__':
    data = DataLoader('raw_data/babyslakh_16k', 'Drums', force=True)
    # print(data.metadata_df)
