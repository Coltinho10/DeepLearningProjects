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
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow import reduce_mean
from sklearn.model_selection import ParameterGrid
from tensorflow_core.python.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout


def get_xy(data_dir, target, preprocess_func):
    """
    Load the raw image data into a data matrix, X, and target vector, y.

    :param data_dir: The root directory of the data
    :param target: The target (subdirectory of the data)
    :param preprocess_func: a function that takes on input 'x' and returns the preprocessed data
    :return: the data matrix 'x' and target vector 'y'
    """
    # load csv file
    df = pd.read_csv(os.path.join(data_dir, target, f'squashpol_{target}.csv'))
    df['image'] = df['image'].str.replace('/', '--')
    df = df.set_index('image')
    x = []
    y = []
    image_files = []
    flowers = []
    date_times = []
    for label in range(2):
        subdir = os.path.join(data_dir, target, str(label))
        for file_name in os.listdir(subdir):
            row = df.loc[file_name]
            image_file = os.path.join(subdir, file_name)
            im = np.array(load_img(image_file))
            x.append(im)
            y.append(label)
            image_files.append(image_file)
            flowers.append(row['flower'])
            date_times.append(row['datetime'])
            assert row['label'] == label, "Something's wrong: subdirectory label doesn't match CSV"

    x = np.array(x)
    y = np.array(y)
    x = preprocess_func(x)
    return x, y, image_files, flowers, date_times


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

    x, y, image_files, flowers, date_times = get_xy(data_dir, target, preprocess_func)
    y_hat = model.predict(x).reshape((-1,))

    loss = reduce_mean(binary_crossentropy(y_true=y, y_pred=y_hat)).numpy()
    acc = reduce_mean(binary_accuracy(y_true=y, y_pred=y_hat)).numpy()

    y_hat_2 = np.maximum(1e-7, np.minimum(1-1e-7, y_hat))
    loss_per_image = -(np.multiply(y, np.log(y_hat_2 + 1e-7), dtype=np.float32) + np.multiply(1-y, np.log(1 - y_hat_2 + 1e-7), dtype=np.float32))

    # update the params dictionary to include train/validation performance
    d2 = {
        f'{target}_loss': float(loss),
        f'{target}_acc': float(acc),
    }
    y_hat_int = (y_hat > 0.5).astype(int)
    # image grid of misclassified examples
    for s in ['fp', 'fn']:
        if s == 'fp':
            # false positives
            index = y_hat_int > y
        elif s == 'fn':
            # false negatives
            index = y_hat_int < y
        else:
            raise ValueError(f'Unknown mistake: {s}')
        # print(s, ':', image_files[index])
        mistakes = x[index]
        d2[f'{target}_{s}'] = len(mistakes)
        if len(mistakes) == 0:
            continue

        mistakes_loss = loss_per_image[index]
        num_mistakes = len(mistakes)
        index = np.random.choice(range(len(mistakes)), size=min(16, len(mistakes)), replace=len(mistakes) < 16)
        mistakes = mistakes[index]
        mistakes_loss = mistakes_loss[index]
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
                ax.set_title(str(mistakes_loss[i]))

        png_file = os.path.join(output_path, f'{target}_{num_mistakes}_{s}_{loss:.6f}_{100*acc:.1f}.png')
        plt.savefig(png_file)
        plt.close(fig)
    print(d2)
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


def assemble_results(output_root, data_dir):
    """
    Helper function to traverse output root directory to assemble and save a CSV file with results.

    :param output_root: The directory that contains all the output model directories
    :param data_dir: The directory containing the "valid" and "train" directories
    :return: None
    """
    data = []
    for run in os.listdir(output_root):
        output_path = os.path.join(output_root, run)
        if os.path.isdir(output_path):
            r = {'dir': run}
            params_file = os.path.join(output_path, 'params.json')
            if os.path.isfile(params_file):
                with open(params_file, 'r') as fp:
                    r.update(flatten(json.load(fp)))
            else:
                continue

            results = get_results(output_path, data_dir)
            r.update(flatten(results))
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

    x, y, image_files, flowers, date_times = get_xy(data_dir, 'valid', preprocess_func)
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
def preprocess(x, target_size, crop, contrast, gamma, hue):
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

    x = tf.image.central_crop(x, crop)  # change target_size
    # x = tf.image.adjust_contrast(x, contrast)
    # x = tf.image.adjust_gamma(x, gamma)
    # x = tf.image.adjust_hue(x, hue)
    # x = tf.image.rgb_to_grayscale(x)
    # x = tf.image.random_hue(x, 0.5)
    # x = tf.image.random_saturation(x, 1, 5)
    # x = tf.image.random_brightness(x, 1, 5)
    # x = tf.image.random_contrast(x, 1, 5)
    # x = tf.image.random_crop(x, (120, 120), 1)
    x = tf.image.resize(x, size=target_size).numpy()

    # make sure to use .numpy() on tensorflow objects to get the numpy array

    return x


# TODO: Modify this function to build your network. Add as many hyperparameters as you like.
def build_fn(input_shape, optimizer, loss, output_activation, metrics, neurons, dropout, learning_rate):
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

    # Fully connected linear model
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',
                     input_shape=input_shape)) # , kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu')) #, kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(neurons, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation=output_activation))
    model.summary()
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


# TODO: Change the way your best model is selected on web-cat by changing "model_selection_quality"
def get_results(output_path, data_dir):
    """
    Return a results dictionary containing performance and "model_selection_quality"
    :param output_path: The directory containing the model and params.
    :param data_dir: The data directory with "train" and "valid" subfolders.
    :return:
    """

    results_file = os.path.join(output_path, 'results.json')
    if os.path.isfile(results_file):
        with open(results_file, 'r') as fp:
            results = json.load(fp)
    else:
        print(f'computing results for {output_path}')
        # get and save results
        results = {}
        preprocess_func = get_preprocess(output_path)
        for target in ['train', 'valid']:
            results.update(analyze_results(output_path, data_dir, target, preprocess_func))

        # you can choose how your models are selected as "best"
        # web-cat will select a user's and a group's best model this way.
        # there will be no section comparisons, so groups don't need to agree on how to measure "best"
        results['model_selection_quality'] = -results['valid_loss']

        with open(results_file, 'w') as fp:
            json.dump(results, fp)

    return results


def main():
    """
    Example of how to do a grid search. Each process gets its own parameter grid.
    Itertools is used to pull one parameter set (dictionary) from each of those.

    :return: None
    """
    # paths
    # data_dir = '/media/music/squashpol_small'
    # output_root = 'squashpol_linear'
    data_dir = './media/music/squashpol_small'
    output_root = './squashpol_linear'

    # dictionary of parameter grids, one for each process
    param_grids = {
        # passed to your preprocess function
        'preprocess_params': ParameterGrid({
            'target_size': [(120, 120)],
            'crop': [0.8],
            'contrast': [1],
            'gamma': [0.1],
            'hue': [1]
        }),
        # passed to the ImageDataGenerator
        'augmentation_params': ParameterGrid({
            'rescale': [1./255],
            'width_shift_range': [1],
            'height_shift_range': [0.2],
            'horizontal_flip': [True],
            'vertical_flip': [True]
        }),
        # passed to your build_fn function
        'model_params': ParameterGrid({
            'learning_rate': [0.001],
            'optimizer': ['adam'],
            'neurons': [512],
            'dropout': [0.4]
        }),
        # passed to the ImageDataGenerator.flow function
        'generator_params': ParameterGrid({
            'batch_size': [32],
        }),
        # passed to the EarlyStopping callback
        'early_stopping_params': ParameterGrid({
            'patience': [10],
        }),
        # passed to the model.fit function
        'fit_params': ParameterGrid({
            'epochs': [1],
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
        x_train, y_train, image_files_train, flowers_train, date_times_train = get_xy(
            data_dir, 'train', preprocess_func
        )
        x_valid, y_valid, image_files_valid, flowers_valid, date_times_valid = get_xy(
            data_dir, 'valid', preprocess_func
        )

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

        try:
            results = get_results(output_path, data_dir)
        except Exception as e:
            print(f'Failed to get results for {output_path}')

        # save a file with name that shows validation performance (for convenience)
        with open(os.path.join(output_path, f'{results["valid_loss"]}_{results["valid_acc"]}.out'), 'w') as fp:
            pass

        try:
            check_model(data_dir, output_path)
        except AssertionError as e:
            print(f'Model check failed: {str(e)}')

    # assemble results from all runs into one CSV file in output root.
    assemble_results(output_root, data_dir)

    # select the best model, zip it's directory and submit to web-cat.


if __name__ == '__main__':
    main()