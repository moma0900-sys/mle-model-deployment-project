from pydantic import BaseModel, Field
from fastapi import FastAPI
import pandas as pd

from pipeline import load_pipeline, predict_duration


''' Beschreibt die erwarteten Fahrtdaten'''
class RideInput(BaseModel):
    PULocationID: int = Field(gt=0)                 # gt: > 0
    DOLocationID: int = Field(gt=0)                 # gt: > 0
    trip_distance: float = Field(ge=0.5, le=20)     # ge: >= 0.5 ... le: <= 20


''' Ist die Webanwendung, an der anschließend /predict anmeldet'''
app = FastAPI()     #TERMINAL- COMMAND: .\.venv\Scripts\python.exe -m uvicorn api:app --reload

''' Tuned model saved in mlflow'''
model_uri = "models:/m-fb68057de49c410eb504e957c141e40d"

''' Load tuned model once as API starts'''
model = load_pipeline(model_uri)

'''
    @app.post("/predict") verbindet die Funktion mit dieser Route.
    ride: RideInput sorgt dafür, dass FastAPI die eingehenden Daten anhand deiner Klasse prüft. Ungültige Eingaben werden abgewiesen.
'''
@app.post("/predict")           #Aufruf bei Post: '/Predict'- Anfrage, übermitteln an API mittels Decorator
def predict(ride:RideInput):
    ride_data = pd.DataFrame([ride.model_dump()])     # .model_dump() wandelt das 'RideInput'-Objekt in ein Dictionary um

    predictions = predict_duration(model, ride_data)
    predicted_duration_minutes = {"predicted_duration_minutes": float(predictions[0])}
    return predicted_duration_minutes