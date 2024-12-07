import io
import os
import keras
import struct
import base64
import numpy as np
from PIL import Image


def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images


def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} for labels file.")
        labels = np.fromfile(f, dtype=np.uint8)
        if len(labels) != num:
            raise ValueError(f"Mismatch: Expected {num} labels, but found {len(labels)}.")
    return labels


def preprocess_data():
    """
    load and preprocess MNIST images.
    """
    base = os. getcwd()
    x_train = load_mnist_images(f'{base}/dataset/train-images.idx3-ubyte')
    y_train = load_mnist_labels(f'{base}/dataset/train-labels.idx1-ubyte')
    x_test = load_mnist_images(f'{base}/dataset/t10k-images.idx3-ubyte')
    y_test = load_mnist_labels(f'{base}/dataset/t10k-labels.idx1-ubyte')

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def decode_base64_image(base64_data):
    """Decode base64 image data into a NumPy array."""
    try:
        return np.array(Image.open(io.BytesIO(base64.b64decode(base64_data))))
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


def preprocess_image(image_array, threshold=200):
    """
    Set the background of an image to white based on a brightness threshold.
    - image_array: the input image as a NumPy array (shape: [height, width, 3] or grayscale).
    - threshold: pixel brightness above which it is considered part of the background.
    """
    if image_array.ndim == 3:
        brightness = np.mean(image_array, axis=-1)
    else:
        brightness = image_array

    image_array[brightness > threshold] = 255

    return image_array


def resize_image(image, size=(28, 28)):
    """Resize the image to the specified size."""
    try:
        return image.resize(size)
    except Exception as e:
        print(f"Error resizing image: {e}")
        return None


def convert_image_to_grayscale(image_array):
    """Convert a colored image to grayscale."""
    try:
        return np.mean(image_array, axis=-1) if image_array.ndim == 3 else image_array
    except Exception as e:
        print(f"Error converting image to grayscale: {e}")
        return None


def prepare_image_for_model(image_array):
    """Prepare the image by resizing and normalizing it for model input."""
    try:
        image_array = np.array(image_array) / 255.0
        return np.expand_dims(image_array, axis=-1)
    except Exception as e:
        print(f"Error preparing image for model: {e}")
        return None
