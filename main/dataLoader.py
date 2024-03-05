import pandas as pd
import yaml
from pathlib import Path
import tensorflow as tf
import numpy as np
import librosa

def configure_for_performance(ds, batch_size, mode):
    ds = ds.cache()
    if mode == 'train':
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

def configure_for_performance(ds, batch_size, mode):
    ds = ds.cache()
    if mode == 'train':
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

class training_DataLoader():
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
        first = True
        for index, row in self.metadata_df.iloc[indexes].iterrows():
            mix_path, instr_path = row['mix_path'], row['instr_path_0']

            y_mix, sr = librosa.load(mix_path)
            mean, stddev = np.mean(y_mix), np.std(y_mix)
            y_mix = (y_mix-mean)/stddev

            y_instr, sr = librosa.load(instr_path)
            mean, stddev = np.mean(y_instr), np.std(y_instr)
            y_instr = (y_instr-mean)/stddev

            if first:
                X_slices = tf.signal.frame(y_mix,
                                          2**15,
                                          2**13,
                                          pad_end=True,
                                          pad_value=0,
                                          axis=-1)
                y_slices = tf.signal.frame(y_instr,
                                          2**15,
                                          2**13,
                                          pad_end=True,
                                          pad_value=0,
                                          axis=-1)

                first = False

            else:
                frames_mix = tf.signal.frame(y_mix,
                                            2**15,
                                            2**14,
                                            pad_end=True,
                                            pad_value=0,
                                            axis=-1)

                X_slices = np.concatenate([X_slices,frames_mix], axis = 0)

                frames_instr = tf.signal.frame(y_instr,
                                              2**15,
                                              2**14,
                                              pad_end=True,
                                              pad_value=0,
                                              axis=-1)

                y_slices = np.concatenate([y_slices,frames_instr], axis = 0)


        X = tf.data.Dataset.from_tensor_slices(X_slices)
        y = tf.data.Dataset.from_tensor_slices(y_slices)

        return configure_for_performance(tf.data.Dataset.zip((X, y)), self.batch_size, mode)
