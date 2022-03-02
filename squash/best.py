import inspect
import itertools
import importlib
import os
import json
import datetime
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import tensorflow
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow import reduce_mean
from sklearn.model_selection import ParameterGrid


def get_xy(data_dir, target, preprocess_func):
    """
    Load the raw image data into a data matrix, X, and target vector, y.

    :param data_dir: The root directory of the data
    :param target: The target (subdirectory of the data)
    :param preprocess_func: a function that takes on input 'x' and returns the preprocessed data
    :return: the data matrix 'x' and target vector 'y'
    """
    x = []
    y = []
    image_files = []
    for label in range(2):
        subdir = os.path.join(data_dir, target, str(label))
        for file_name in os.listdir(subdir):
            image_file = os.path.join(subdir, file_name)
            im = np.array(load_img(image_file))
            x.append(im)
            y.append(label)
            image_files.append(image_file)
    x = np.array(x)
    y = np.array(y)
    x = preprocess_func(x)
    return x, y, image_files


def get_h5_file(output_dir='.'):
    for model_file in os.listdir(output_dir):
        if model_file.endswith('.h5') or model_file.endswith('.hdf5'):
            model_file = os.path.join(output_dir, model_file)
            print(f'Found {model_file}')
            break
    else:
        return None
    return load_model(model_file, compile=True)


def analyze_results(output_path, data_dir, target, preprocess_func):
    """
    Helper function to compute train/validation performance and show some of the mistakes.

    :param output_path: path to folder containing trained model
    :param data_dir: where to find the train/validation data
    :param target: 'train' or 'valid'
    :param preprocess_func: the preprocessing function that takes one parameter 'x'
    :return: a dictionary with train/valid loss/accuracy
    """
    clear_session()
    model_file = os.path.join(output_path, 'model.h5')
    # get best model and update JSON with results
    model = load_model(model_file, compile=True)

    x, y, image_files = get_xy(data_dir, target, preprocess_func)
    y_hat = model.predict(x).reshape((-1,))

    loss = reduce_mean(binary_crossentropy(y_true=y, y_pred=y_hat)).numpy()
    acc = reduce_mean(binary_accuracy(y_true=y, y_pred=y_hat)).numpy()

    # update the params dictionary to include train/validation performance
    d2 = {
        f'{target}_loss': float(loss),
        f'{target}_acc': float(acc),
    }
    print(d2)
    y_hat = (y_hat > 0.5).astype(int)
    image_files = np.array([im.split('/')[-1] for im in image_files])
    # image grid of misclassified examples
    for s in ['fp', 'fn']:
        if s == 'fp':
            # false positives
            index = y_hat > y
        elif s == 'fn':
            # false negatives
            index = y_hat < y
        else:
            raise ValueError(f'Unknown mistake: {s}')
        print(s, ':', image_files[index])
        mistakes = x[index]
        num_mistakes = len(mistakes)
        index = np.random.choice(range(len(mistakes)), size=min(16, len(mistakes)), replace=len(mistakes) < 16)
        mistakes = mistakes[index]
        fig = plt.figure(figsize=(8, 8))
        grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0.1)
        for i, ax in enumerate(grid):
            if i < len(mistakes):
                im = mistakes[i]
                im = im - im.min(axis=(0, 1))
                im = im / im.max(axis=(0, 1))
                if im.shape[2] == 2:
                    im = np.concatenate((im, np.zeros(im.shape[:2] + (1,))), axis=2)
                ax.imshow(im)

        png_file = os.path.join(output_path, f'{target}_{num_mistakes}_{s}_{loss:.6f}_{100 * acc:.1f}.png')
        plt.savefig(png_file)
        plt.close(fig)
    return d2


def flatten(x, sep='.'):
    obj = {}

    def recurse(t, parent_key=''):
        if isinstance(t, list):
            for i, ti in enumerate(t):
                recurse(ti, f'{parent_key}{sep}{i}' if parent_key else str(i))
        elif isinstance(t, dict):
            for k, v in t.items():
                recurse(v, f'{parent_key}{sep}{k}' if parent_key else k)
        else:
            obj[parent_key] = t

    recurse(x)
    return obj


def assemble_results(output_root, param_grid_names):
    """
    Helper function to traverse output root directory to assemble and save a CSV file with results.

    :param output_root: The directory that contains all the output model directories
    :param param_grid_names: The names of the parameter sets stored in JSON files
    :return: None
    """
    data = []
    for run in os.listdir(output_root):
        run_dir = os.path.join(output_root, run)
        if os.path.isdir(run_dir):
            r = {'dir': run}
            for name in ['params', 'results']:
                json_file = os.path.join(run_dir, f'{name}.json')
                try:
                    with open(json_file, 'r') as fp:
                        r.update(flatten(json.load(fp)))
                except (FileNotFoundError, KeyError) as e:
                    print(str(e))
            data.append(r)

    csv_file = os.path.join(output_root, 'results.csv')
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)


def get_preprocess(output_dir='.'):
    spec = importlib.util.spec_from_file_location('preprocess', os.path.join(output_dir, 'squashpol.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = getattr(mod, 'preprocess')

    with open(os.path.join(output_dir, 'params.json'), 'r') as fp:
        params = json.load(fp)
    preprocess_params = params['preprocess_params']

    def f(data):
        q = p(data, **preprocess_params)
        return q.numpy() if hasattr(q, 'numpy') else q

    return f


def check_model(data_dir, output_path):
    clear_session()
    model = get_h5_file(output_path)
    preprocess_func = get_preprocess(output_path)

    x, y, image_files = get_xy(data_dir, 'valid', preprocess_func)
    y_hat = model.predict(x).reshape((-1,))

    loss = reduce_mean(binary_crossentropy(y_true=y, y_pred=y_hat)).numpy()
    acc = reduce_mean(binary_accuracy(y_true=y, y_pred=y_hat)).numpy()

    results_file = os.path.join(output_path, 'results.json')
    with open(results_file, 'r') as fp:
        results = json.load(fp)

    print(f'Model at {output_path}:')
    print(f'\t{"Computed":25s}: loss = {loss:.6f}, acc = {acc:.4%}')
    print(f'\t{"Results file":25s}: loss = {results["valid_loss"]:.6f}, acc = {results["valid_acc"]:.4%}')

    assert abs(loss - results['valid_loss']) < 1e-6, f'Computed loss does not match recorded loss'
    assert abs(acc - results['valid_acc']) < 1e-6, f'Computed accuracy does not match recorded accuracy'


# TODO: Modify this function to preprocess each data set. Add as many hyperparameters as you like.
def preprocess(x, target_size=(10, 10)):
    """
    This is a preprocess function meant to be tuned using input arguments. Once you
    determine the best preprocessing approach, make sure to save the keyword arguments
    in preprocess_params.json to submit to Web-CAT. The example "main" code does this.

    :param x: The data matrix (n, 240, 240, 3) numpy array uint8
    :param target_size: Resize the images to this shape
    :return: The updated data matrix a numpy array (n, num_rows, num_columns, num_channels)
    """
    # make sure any imports you need are imported here.
    import tensorflow as tf

    x = tf.image.central_crop(x, 0.9)
    # x = tf.image.adjust_contrast(x, 2)
    # x = tf.image.adjust_gamma(x, 0.2)
    # x = tf.keras.applications.vgg19.preprocess_input(x)
    # print(x.shape)
    # print(np.mean(x, axis=(0, 1, 2)))
    # print(np.std(x, axis=(0, 1, 2)))
    # x = tf.image.per_image_standardization(x)
    x = tf.cast(x, tf.float32)
    x = tf.image.resize(x, size=target_size).numpy()
    x[..., 0] -= 136.80579317
    x[..., 1] -= 125.27596712
    x[..., 2] -= 73.32884733

    x[..., 0] /= 82.66153096
    x[..., 1] /= 78.18526396
    x[..., 2] /= 71.58871252
    print(x.shape)
    print(np.mean(x, axis=(0, 1, 2)))
    print(np.std(x, axis=(0, 1, 2)))

    # make sure to use .numpy() on tensorflow objects to get the numpy array
    return x


# TODO: Modify this function to build your network. Add as many hyperparameters as you like.
def build_fn(input_shape, optimizer, learning_rate, loss, output_activation, metrics, neurons, dropout):
    """
    Custom model build function can take any parameters you want to build a network
    for your model.

    :param input_shape: the shape of each sample (image)
    :param optimizer: the optimizer function
    :param loss: the loss function
    :param output_activation: the output activation function
    :param metrics: other metrics to track
    :return: a compiled model ready to 'fit'
    """
    from tensorflow.keras.applications.resnet50 import ResNet50

    # make sure to clear any previous nodes in the computation graph to save memory
    clear_session()

    # Load the pre-trained VGG19 model:
    # res = ResNet50(include_top=False,
    #           weights=None,
    #           input_shape=input_shape,
    #           pooling=None)

    # Freeze all the layers in the base VGGNet19 model:
    # for layer in res.layers:
    #     layer.trainable = False
    # vgg.get_layer('block1_conv1').trainable = False
    # vgg.get_layer('block1_conv2').trainable = False
    # vgg.get_layer('block2_conv1').trainable = False
    # vgg.get_layer('block2_conv2').trainable = False
    # vgg.get_layer('block3_conv1').trainable = False
    # vgg.get_layer('block3_conv2').trainable = False
    # vgg.get_layer('block3_conv3').trainable = False
    # vgg.get_layer('block3_conv4').trainable = False

    # res.summary()

    # optimizer = tensorflow.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=False, name="SGD")

    # Fully connected linear model
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',
                     input_shape=input_shape, kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())

    model.add(
        Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())

    model.add(
        Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())

    model.add(
        Conv2D(512, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    # model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(neurons, activation='relu', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation=output_activation))
    model.summary()
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def main():
    """
    Example of how to do a grid search. Each process gets its own parameter grid.
    Itertools is used to pull one parameter set (dictionary) from each of those.

    :return: None
    """
    # paths
    data_dir = './media/music/squashpol_small'
    output_root = './squashpol_linear'

    # dictionary of parameter grids, one for each process
    param_grids = {
        # passed to your preprocess function
        'preprocess_params': ParameterGrid({
            'target_size': [(120, 120)],  # (192, 192)
        }),
        # passed to the ImageDataGenerator
        'augmentation_params': ParameterGrid({
            'width_shift_range': [1],
            'height_shift_range': [0.2],
            'horizontal_flip': [True],
            'vertical_flip': [True]
        }),
        # passed to your build_fn function
        'model_params': ParameterGrid({
            'learning_rate': [0.001],
            'optimizer': ['nadam'],
            'neurons': [512],
            'dropout': [0.3]
        }),
        # passed to the ImageDataGenerator.flow function
        'generator_params': ParameterGrid({
            'batch_size': [32],
        }),
        # passed to the EarlyStopping callback
        'early_stopping_params': ParameterGrid({
            'patience': [30],
        }),
        # passed to the model.fit function
        'fit_params': ParameterGrid({
            'epochs': [500],
        }),
    }

    # create list of names and corresponding parameter grids for use with itertools.product
    param_grid_names = list(param_grids.keys())
    param_grid_list = [param_grids[k] for k in param_grid_names]

    for params in itertools.product(*param_grid_list):
        # store parameters in dictionary, one item per process
        params = {k: v for k, v in zip(param_grid_names, params)}
        print('params:', params)

        # setup output directory
        date = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        output_path = os.path.join(output_root, f'{date}')
        os.makedirs(output_path, exist_ok=True)

        # make the single argument preprocess function
        def preprocess_func(x):
            return preprocess(x, **params['preprocess_params'])

        # load data and preprocess it with preprocess parameters
        x_train, y_train, image_files_train = get_xy(data_dir, 'train', preprocess_func)
        x_valid, y_valid, image_files_valid = get_xy(data_dir, 'valid', preprocess_func)

        # create training data generator with generator parameters
        train_generator = ImageDataGenerator(**params['augmentation_params']).flow(x_train, y_train,
                                                                                   **params['generator_params'])

        # build model with model parameters
        # input_shape must match the shape output by your preprocess function
        model = build_fn(input_shape=x_valid.shape[1:], output_activation='sigmoid', metrics=['accuracy'],
                         loss='binary_crossentropy', **params['model_params'])

        # setup early stopping with early stopping parameters
        early_stopping = EarlyStopping(monitor='val_loss', verbose=1, **params['early_stopping_params'])
        # setup model checkpointing
        model_checkpoint = ModelCheckpoint(
            filepath=os.path.join(output_path, 'model.h5'),  # always overwrite the existing model
            save_weights_only=False, save_freq='epoch',
            save_best_only=True, monitor='val_loss', verbose=1)  # only save models that improve the 'monitored' value
        callbacks = [early_stopping, model_checkpoint]

        # update fit params based on batch size
        # when using an ImageDataGenerator, model.fit needs to know how many batches to use per epoch
        params['fit_params'].update(dict(
            steps_per_epoch=len(x_train) // params['generator_params']['batch_size'],
        ))

        # save parameters to output path
        with open(os.path.join(output_path, f'params.json'), 'w') as fp:
            json.dump(params, fp)

        # save a copy of *this* Python file, including build_fn and preprocess.
        shutil.copyfile(__file__, os.path.join(output_path, 'squashpol.py'))

        # train model
        model.fit(train_generator, validation_data=(x_valid, y_valid), callbacks=callbacks, verbose=1,
                  **params['fit_params'])

        # get and save results
        results = {}
        for target in ['train', 'valid']:
            results.update(analyze_results(output_path, data_dir, target, preprocess_func))

        # you can choose how your models are selected as "best"
        # web-cat will select a user's and a group's best model this way.
        # there will be no section comparisons, so groups don't need to agree on how to measure "best"
        results['model_selection_quality'] = -results['valid_loss']

        with open(os.path.join(output_path, 'results.json'), 'w') as fp:
            json.dump(results, fp)

        # save a file with name that shows validation performance (for convenience)
        with open(os.path.join(output_path, f'{results["valid_loss"]}_{results["valid_acc"]}.out'), 'w') as fp:
            pass

        check_model(data_dir, output_path)

    # assemble results from all runs into one CSV file in output root.
    assemble_results(output_root, param_grid_names)

    # select the best model, zip it's directory and submit to web-cat.


if __name__ == '__main__':
    main()