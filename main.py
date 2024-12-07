import os
from services.model_service import start_training, predict_image, training_status
from schemas import TrainingRequest, PredictionRequest, PredictionResponse
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from keras.src.saving import load_model



app = FastAPI()

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def get_ui(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/train")
async def train(request: TrainingRequest):
    """
    request to start training.
    """
    if not os.path.exists('mnist_classifier.h5'):
        try:
            train_uuid = start_training(request.epochs, request.batch_size)
            return JSONResponse(content={"uuid": train_uuid})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail="trained model exist.")


@app.get("/train_status/{train_uuid}")
async def get_training_status(train_uuid: str):
    """
    get training status by uuid.
    """
    status = training_status.get(train_uuid, "No such training found")
    return JSONResponse(content={"status": status})


@app.post("/predict/")
async def predict(request: PredictionRequest):
    """
    prediction endpoint by traning model.
    """
    if os.path.exists('mnist_classifier.h5'):
        model = load_model('mnist_classifier.h5')

        predicted_label, inference_time = predict_image(request.image, model)
        return PredictionResponse(predicted_label=predicted_label, inference_time=inference_time)
    else:
        raise HTTPException(status_code=404, detail='The model has not been trained yet. Please start the training process first.')

