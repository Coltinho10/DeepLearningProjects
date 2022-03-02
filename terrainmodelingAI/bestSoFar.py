
from terrain_helpers import *
from keras.initializers import glorot_normal, Zeros
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Lambda
from keras.layers import Dropout
from tensorflow.keras.layers import Activation, Dropout, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
# from keras.layers.normalization import BatchNormalization
output_path = 'terrain/linear'
tiff_file = f'terrain/Grand_Canyon_0.1deg.tiff'
model_file = f'{tiff_file}.h5'
# model_file = 'currentFile9_11_GC_regular.h5'

model_checkpoint = setup_model_checkpoints(output_path, save_freq='epoch')
x, y = get_xy(tiff_file)

keras.backend.clear_session()
b_init = Zeros()
w_init = glorot_normal()
# linear regression


# model = Sequential()
model = load_model('terrain/Grand_Canyon_0.1deg.tiff.h5')

model.fit(x, y, batch_size=128, verbose=1, epochs=10, callbacks=[model_checkpoint])
compare_images(model, x, y, output_path)

# model_file = f'{tiff_file}.h5'
model.save(model_file)