from gc import callbacks
from lib2to3.pgen2.token import ENDMARKER
from tabnanny import verbose
from tkinter import FALSE
from venv import create
import tensorflow as tf

from sklearn import metrics
from utils_functions import order_test_set, split_data, create_generator
from deeplearning_model import functional_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

if __name__=="__main__":

    data_to_copy = False
    if data_to_copy:
        path_to_data = "/home/bishesh/Desktop/Traffic Sign/data/train"
        path_to_save_train = "/home/bishesh/Desktop/Traffic Sign/data/training_data/train"
        path_to_save_val = "/home/bishesh/Desktop/Traffic Sign/data/training_data/val"
        split_data(path_to_data=path_to_data, path_to_save_train=path_to_save_train, path_to_save_val=path_to_save_val)

    
    test_file_to_arrange = False
    if test_file_to_arrange:
        path_to_images = "/home/bishesh/Desktop/Traffic Sign/data/test"
        path_to_csv = "/home/bishesh/Desktop/Traffic Sign/data/Test.csv"
        order_test_set(path_to_images=path_to_images, path_to_csv=path_to_csv)


    path_to_train = "/home/bishesh/Desktop/Traffic Sign/data/training_data/train"
    path_to_val = "/home/bishesh/Desktop/Traffic Sign/data/training_data/val"
    path_to_test = "/home/bishesh/Desktop/Traffic Sign/data/test"    
    batch_size = 64
    epochs = 15

    train_gen, val_gen, test_gen = create_generator(batch_size, train_data_path=path_to_train, val_data_path=path_to_val, test_data_path=path_to_test)
    n_class = train_gen.num_classes

    TRAIN = False
    TEST = FALSE

    if TRAIN:
        path_to_save_model = './Models'
        ckpoint_saver = ModelCheckpoint(
            path_to_save_model,
            monitor = "val_accuracy",
            mode = 'max',
            save_best_only = True,
            save_freq = 'epoch',
            verbose = 1
        )

        early_stop = EarlyStopping(monitor='val_accuracy', patience=10)

        model = functional_model(n_class)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(train_gen,
                epochs = epochs,
                batch_size = batch_size,
                validation_data = val_gen,
                callbacks = [ckpoint_saver, early_stop]
                )

    if TEST:
        model = tf.keras.models.load_model("./Models")
        model.summary()

        print("Evaluating Validation Set:")
        model.evaluate(val_gen)

        print("Evaluating Test Set:")
        model.evaluate(test_gen)