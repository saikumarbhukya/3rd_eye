from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import httpx
import uvicorn
from typing import List
import os

app = FastAPI(title="Object Detection UI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DetectionResponse:
    def __init__(self, objects: List[dict], processing_time: float):
        self.objects = objects
        self.processing_time = processing_time

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    # Read the image file
    contents = await file.read()
    
    # Send the image to AI service
    async with httpx.AsyncClient() as client:
        files = {'file': contents}
        response = await client.post(
            'http://ai-service:5000/process',
            files=files
        )
        return response.json()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
