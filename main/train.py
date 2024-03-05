from dataLoader import *
from Unet_Model import *
import tensorflow as tf
import keras.backend as K
import gc
from sklearn.model_selection import KFold

# USE MULTIPLE GPUS
gpus = tf.config.list_physical_devices('GPU')
if len(gpus)<=1:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print(f'Using {len(gpus)} GPU')
else:
    strategy = tf.distribute.MirroredStrategy()
    print(f'Using {len(gpus)} GPUs')

EPOCHS = 5
n_folds = 5
instr_class = 'Drums'

dataset = training_DataLoader('/content/drive/MyDrive/babyslakh', instr_class)

gkf = KFold(n_splits=n_folds)

for i, (train_index, valid_index) in enumerate(gkf.split(dataset.metadata_df)):

    print('#'*25)
    print(f'### Fold {i+1}')
    print('#'*25)

    train_ds = dataset.get_data(train_index, 'train')
    val_ds = dataset.get_data(valid_index, 'test')

    with strategy.scope():
        model = build_unet_stft_model()

    model.fit(train_ds, verbose=1,
              validation_data = val_ds,
              epochs=EPOCHS,
              batch_size=8)#, callbacks = [LR] )
    model.save_weights(f'Unet_{instr_class}_f{i}.h5')

    K.clear_session()
    del model
    del train_ds
    del val_ds
    gc.collect()
