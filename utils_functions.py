import os
import glob
import shutil
import csv
from tkinter import Image
from sklearn import model_selection
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def split_data(path_to_data, path_to_save_train, path_to_save_val, split_size=0.1):

    folders = os.listdir(path_to_data)

    for folder in folders:

        full_path = os.path.join(path_to_data, folder)
        img_paths = glob.glob(os.path.join(full_path, '*.png'))

        x_train, x_val = model_selection.train_test_split(img_paths, test_size=split_size)

        for x in x_train:

            path_to_folder = os.path.join(path_to_save_train, folder)

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x, path_to_folder)

        for x in x_val:

            path_to_folder = os.path.join(path_to_save_val, folder)

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x, path_to_folder)


def order_test_set(path_to_images, path_to_csv):

    testset = { }

    try:
        with open(path_to_csv, 'r') as csvfile:

            reader = csv.reader(csvfile, delimiter=',')

            for i, row in enumerate(reader):

                if i==0:
                    continue

                img_name = row[-1].replace('Test/', '')
                label = row[-2]

                path_to_folder = os.path.join(path_to_images, label)

                if not os.path.isdir(path_to_folder):
                    os.makedirs(path_to_folder)

                img_full_path = os.path.join(path_to_images, img_name)
                shutil.move(img_full_path, path_to_folder)

    except:
        print('[INFO] : Error reading csv file')


def create_generator(batch_size, train_data_path, val_data_path, test_data_path):

    train_preprocessor = ImageDataGenerator(
        rescale = 1 / 255.,
        rotation_range = 10,
        width_shift_range = 0.1
    )

    test_preprocessor = ImageDataGenerator(
        rescale = 1 / 255.
    )

    train_generator = train_preprocessor.flow_from_directory(
        train_data_path,
        class_mode = 'categorical',
        target_size = (60,60),
        color_mode = 'rgb',
        shuffle = True,
        batch_size = batch_size
    )

    val_generator = test_preprocessor.flow_from_directory(
        val_data_path,
        class_mode = 'categorical',
        target_size = (60,60),
        color_mode = 'rgb',
        shuffle = False,
        batch_size = batch_size
    )

    test_generator = test_preprocessor.flow_from_directory(
        test_data_path,
        class_mode = 'categorical',
        target_size = (60,60),
        color_mode = 'rgb',
        shuffle = False,
        batch_size = batch_size
    )

    return train_generator, val_generator, test_generator
