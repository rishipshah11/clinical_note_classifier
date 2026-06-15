from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict

app = FastAPI(title="Clinical Note Classifier API")

class NoteRequest(BaseModel):
    text: str

class NoteResponse(BaseModel):
    label: str
    emoji: str
    confidence: float
    all_classes: dict

@app.post("/classify", response_model=NoteResponse)
def classify(req: NoteRequest):
    return predict(req.text)

@app.get("/health")
def health():
    return {"status": "ok"}
