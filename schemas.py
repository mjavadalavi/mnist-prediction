from pydantic import BaseModel


class TrainingRequest(BaseModel):
    epochs: int
    batch_size: int


class PredictionRequest(BaseModel):
    image: str  # Base64 encoded image data


class PredictionResponse(BaseModel):
    predicted_label: int
    inference_time: float
