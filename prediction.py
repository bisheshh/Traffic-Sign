import tensorflow as tf
import numpy as np



def predict_with_model(model, imgpath):

    image = tf.io.read_file(imgpath)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, [70,70]) #(60,60,3)
    image = tf.expand_dims(image, axis=0)   #(1,60,60,3)

    predictions = model.predict(image) # [0.05, 0.006, 0.99, 0.001 ....]

    predictions = np.argmax(predictions) # 2

    return predictions


if __name__ == "__main__":

    model = tf.keras.models.load_model("./Models")
    img_path = "/home/bishesh/Desktop/Traffic Sign/data/test/2/00034.png"
    img_path = "/home/bishesh/Desktop/Traffic Sign/data/test/16/00000.png"
    pred = predict_with_model(model, imgpath=img_path)

    print(f"prediction = {pred}")

    


