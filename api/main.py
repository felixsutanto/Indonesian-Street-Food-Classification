# api/main.py
import io
import os
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf

# --- Application and Model Setup ---
app = FastAPI(title="Indonesian Food Classifier", version="1.0")

# This path is relative to the WORKDIR inside the Docker container
MODEL_PATH = "models/indonesian_food_cnn.h5"
model = None
# The class order MUST match the training output
CLASS_NAMES = ['bakso', 'martabak', 'sate'] 
IMG_HEIGHT = 224
IMG_WIDTH = 224

@app.on_event("startup")
def load_model():
    """Load the model when the API starts."""
    global model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"ERROR: Model file not found at {MODEL_PATH}")

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocesses the image for the model."""
    if image.mode != "RGB": image = image.convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

@app.get("/", tags=["General"])
def read_root():
    return {"message": "Welcome to the Indonesian Food Classifier API. See /docs for usage."}

@app.get("/health", tags=["General"])
def health_check():
    """Check if the API and model are ready."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or not found.")
    return {"status": "ok", "model_loaded": True}

@app.post("/predict", tags=["Classification"])
async def predict(file: UploadFile = File(...)):
    """Accepts an image file and returns the predicted food class."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        processed_image = preprocess_image(image)
        prediction_scores = model.predict(processed_image)[0]
        
        predicted_index = np.argmax(prediction_scores)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(prediction_scores[predicted_index])
        
        return {
            "prediction": predicted_class,
            "confidence": f"{confidence:.2%}",
            "scores": {name: f"{float(score):.2%}" for name, score in zip(CLASS_NAMES, prediction_scores)}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")
