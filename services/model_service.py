import os
import time
import uuid
import threading
import numpy as np
from PIL import Image
from keras import layers, Sequential, Input
from utils.data_processing import preprocess_data, decode_base64_image, preprocess_image, resize_image, \
    convert_image_to_grayscale, prepare_image_for_model

num_classes = 10
input_shape = (28, 28, 1)

model = Sequential([
    Input(shape=input_shape),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax"),
])

training_status = {}


def train_model(epochs: int, batch_size: int, train_uuid: str):
    """
    training model with MNIST dataset
    """
    x_train, y_train, x_test, y_test = preprocess_data()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    for epoch in range(epochs):
        if train_uuid not in training_status:
            return 
        model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_split=0.2, verbose=0)
        training_status[train_uuid] = f"Epoch {epoch + 1} of {epochs} completed"
        time.sleep(2)
    
    model.save(f'{os.getcwd()}/mnist_classifier.h5')
    training_status[train_uuid] = "Training finished"


def start_training(epochs: int, batch_size: int):
    """
    training model in new thread
    """
    train_uuid = str(uuid.uuid4())
    training_status[train_uuid] = "Training started"

    threading.Thread(target=train_model, args=(epochs, batch_size, train_uuid)).start()

    return train_uuid


def predict_image(base64_data, model):
    start_time = time.time()

    # 1. Decode the image
    image_array = decode_base64_image(base64_data)
    if image_array is None:
        return None

    # 2. Preprocess the image
    processed_image = preprocess_image(image_array)
    if processed_image is None:
        return None

    # 3. Resize the image
    resized_image = resize_image(Image.fromarray(processed_image))
    if resized_image is None:
        return None

    # 4. Convert to grayscale
    grayscale_image = convert_image_to_grayscale(np.array(resized_image))
    if grayscale_image is None:
        return None

    # 5. Prepare image for model prediction
    prepared_image = prepare_image_for_model(grayscale_image)
    if prepared_image is None:
        return None

    # 6. Model prediction
    prediction = model.predict(np.expand_dims(prepared_image, axis=0))
    predicted_label = np.argmax(prediction)

    end_time = time.time()
    inference_time = end_time - start_time

    return predicted_label, inference_time
