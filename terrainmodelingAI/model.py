
from terrain_helpers import *
from keras.initializers import glorot_normal, Zeros
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Lambda
from keras.layers import Dropout
from tensorflow.keras.layers import Activation, Dropout, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
# from keras.layers.normalization import BatchNormalization
output_path = 'terrain/linear'
tiff_file = f'terrain/Grand_Canyon_1.0deg.tiff'
model_file = f'{tiff_file}.h5'
# model_file = 'currentFile9_11_GC_regular.h5'

model_checkpoint = setup_model_checkpoints(output_path, save_freq='epoch')
x, y = get_xy(tiff_file)

keras.backend.clear_session()
b_init = Zeros()
w_init = glorot_normal()
# linear regression

#model = Sequential()
model = load_model('terrain/linear/model.00005_8.6092.h5')
#model.add(Input(2))
#model.add(Dense(64, activation='relu', kernel_initializer=w_init, bias_initializer=b_init))
#model.add(BatchNormalization())
##model.add(Dense(32, activation='relu', kernel_initializer=w_init, bias_initializer=b_init))
##model.add(BatchNormalization())
##model.add(Dense(32, activation='relu', kernel_initializer=w_init, bias_initializer=b_init))
##model.add(BatchNormalization())
##model.add(Dense(32, activation='relu', kernel_initializer=w_init, bias_initializer=b_init))
##model.add(BatchNormalization())
##model.add(Dense(1, activation='linear', kernel_initializer=w_init, bias_initializer=b_init))

#y_mean = y.mean()
#model.add(Lambda(lambda v: v + y_mean))
#print("YMEAN", y_mean)
#model.add(Lambda(lambda v: v + 1846.3201))
# 1348.8816
# 1846.3201
#model.compile(loss='mean_squared_error', optimizer='adam', metrics=[Entropy()])
#model.summary()

#print_error(y, y.mean(), 1, 'Constant')

model.fit(x, y, batch_size=1024, verbose=1, epochs=10, callbacks=[model_checkpoint])
compare_images(model, x, y, output_path)

# model_file = f'{tiff_file}.h5'
model.save(model_file)
#8.04
#10.99
#12.16 terrain/linear/model.00005_8.6161.h5
#12.66 terrain/linear/model.00005_8.6092.h5
#12.35      terrain/linear/model.00005_8.6092.h5 2048 batch size same epochs as before
#10.33           terrain/linear/model.00005_8.6092.h5 512 batch size same epochs as before
# ended with 12.24%