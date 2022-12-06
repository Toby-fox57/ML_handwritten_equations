import os
import cv2
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("number of GPUs available: ", len(tf.config.list_physical_devices('GPU')))

FILE_DIR = 'extracted_images/'
IMAGE_SHAPE = 45


# Toby code
def image_data(input_size=IMAGE_SHAPE, file_directory=FILE_DIR):
    file_names = os.listdir(file_directory)
    file_length, labels, data = np.empty(0), np.empty(0), np.empty((0, input_size, input_size))

    for index, character in enumerate(file_names):

        file_length = np.append(file_length, len(os.listdir(file_directory + character)))
        labels = np.append(target, np.linspace(index, index, file_length[index].astype(int)))

        imgs = np.ndarray((file_length[index].astype(int), 45, 45), dtype='uint8')
        for count, img in enumerate(os.listdir(file_directory + character)):
            imgs[count][:, :] = cv2.imread(file_directory + character + '/' + img, 0)
        data = np.vstack((data, imgs))

        print("Characters", index + 1, "/", len(file_list), "processed")

    return file_length, labels, data


def train_and_test(label, data):
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)

    return x_train, x_test, y_train, y_test


# Wardii code

def calculate_output_filters(input_volume, kernel_size, stride, padding):
    return ((input_volume - kernel_size + 2 * padding) / stride) + 1


def model_builder(number_classes, input_size=IMAGE_SHAPE):
    output_filters = calculate_output_filters(input_size, 3, 1, 0)

    conv1 = tf.keras.layers.Conv1D(output_filters, 3, 1, input_shape=(input_size, input_size))
    output_filters = calculate_output_filters(output_filters, 3, 1, 0)
    dense1 = tf.keras.layers.Dense(32, activation='relu')
    conv2 = tf.keras.layers.Conv1D(output_filters, 3, 1)
    output_filters = calculate_output_filters(output_filters, 3, 1, 0)
    dense2 = tf.keras.layers.Dense(32, activation='relu')
    maxPool = tf.keras.layers.MaxPool1D(2)
    output_filters /= 2
    dense3 = tf.keras.layers.Dense(32, activation='relu')
    conv3 = tf.keras.layers.Conv1D(output_filters, 3, 1)
    output_filters = calculate_output_filters(output_filters, 3, 1, 0)
    dense4 = tf.keras.layers.Dense(32, activation='relu')
    output_filters /= 2

    model = tf.keras.models.Sequential([
        conv1,
        dense1,
        conv2,
        dense2,
        maxPool,
        dense3,
        conv3,
        dense4,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(number_of_classes, activation="softmax")
    ])

    return model


def model_compiler(model):
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    return model


def model_test(model):
    model.fit(x_train, y_train, batch_size=16, epochs=3)  # Accuracy = 97.44%
    model.evaluate(x_test, y_test, batch_size=16)

    model.save('saved_model/my_model')


# RUN CODE

def main():
    file_length, labels, data = image_data()
    x_train, x_test, y_train, y_test = train_and_test(labels, data)

    model = model_builder(file_length.size)
    model = model_compiler(model)
    model_test(model)

    return 0

