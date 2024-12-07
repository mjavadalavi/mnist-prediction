
# MNIST Digit Prediction using FastAPI and Keras

This project provides a web application that allows users to draw digits on a canvas, and the model predicts the digit using a Convolutional Neural Network (CNN). The backend is powered by FastAPI, and the model is trained using Keras with the MNIST dataset.

## Features
- **Digit Drawing**: Draw digits on a canvas.
- **Prediction**: The model predicts the digit in real-time.
- **Training**: Train the model through a REST API endpoint.
- **Status Updates**: Get updates on the training progress using a UUID.

## How It Works

The web application allows users to draw digits on a canvas, which are then processed and passed to a trained CNN model. The backend is powered by FastAPI and uses Keras for training and predicting the model. The status of model training can also be queried with a unique UUID.

### Key Components:
- **FastAPI**: Web framework for serving the application and model.
- **Keras**: Deep learning framework used to build and train the CNN model.
- **MNIST Dataset**: The dataset used for training the model.
- **Canvas**: An HTML5 canvas element used to draw the digits.

## Requirements

Before running the application, ensure you have the following dependencies installed:

1. **Python 3.x**
2. **Dependencies**:
   - `FastAPI`
   - `Uvicorn`
   - `Keras`
   - `TensorFlow`
   - `Numpy`
   - `Pillow`
   -  `jinja2`

You can install the required dependencies by running:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```txt
fastapi
uvicorn
keras
tensorflow
numpy
pillow
jinja2
```

## Running the Application

To run the FastAPI application locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/mjavadalavi/mnist-prediction.git
   cd mnist-prediction
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the FastAPI application:

   ```bash
   uvicorn main:app --reload
   ```

4. Open your browser and navigate to `http://127.0.0.1:8000` to access the application.

## API Endpoints

### 1. **Index Page**

- **Endpoint**: `Get /`


### 2. **Train the Model**

- **Endpoint**: `POST /train`
- **Request Body**: 
  ```json
  {
    "epochs": 10,
    "batch_size": 128
  }
  ```

- **Response**: 
  ```json
  {
    "uuid": "unique-training-uuid"
  }
  ```

### 3. **Get Training Status**

- **Endpoint**: `GET /train_status/{train_uuid}`
- **Response**: 
  ```json
  {
    "status": "Training started" // or "Training finished"
  }
  ```

### 4. **Predict Digit**

- **Endpoint**: `POST /predict/`
- **Request Body**: 
  ```json
  {
    "image": "<base64-encoded-image>"
  }
  ```

- **Response**:
  ```json
  {
    "predicted_label": 5,
    "inference_time": 0.2
  }
  ```

## Demo

### Here's a quick demonstration of how the application works:

1. Train the model.
2. Draw a digit on the canvas.
3. Click the **Predict** button.
4. Wait for the prediction result to appear.

## Project Structure

The project is structured as follows:

```
project/
├── main.py                # FastAPI application (routes and logic)
├── services/
│   └── model_service.py   # Model training and prediction logic
├── utils/
│   └── data_processing.py # Data loading and processing functions
├── schemas.py             # Pydantic models for request and response validation
├── requirements.txt       # List of Python dependencies
└── run.sh                 # Script to run the application
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
