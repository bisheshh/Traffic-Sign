from turtle import shape
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D,  BatchNormalization, Dense, GlobalAvgPool2D
from tensorflow.keras import Model

def functional_model(n_classes):

    my_input = Input(shape=(60,60,3))

    x = Conv2D(32, (3,3), activation='relu')(my_input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = GlobalAvgPool2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(n_classes, activation='softmax')(x)

    return Model(inputs=my_input, outputs=x)

if __name__ == "__main__":

    model = functional_model(10)
    model.summary()