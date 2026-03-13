import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.workflow.pipeline import run_pipeline

app = FastAPI(
    title="Video Object Counting API",
    description="Count objects, actions, or extract numbers from YouTube videos.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    video_id: str
    question: str


class QueryResponse(BaseModel):
    video_id: str
    question: str
    task: str
    answer: str


@app.get("/")
def root():
    return {"message": "Video Object Counting API is running."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=QueryResponse)
def predict(request: QueryRequest):
    """Run the full pipeline for a given YouTube video and question."""
    try:
        result = run_pipeline(request.video_id, request.question)
        return QueryResponse(
            video_id=result["video_id"],
            question=result["question"],
            task=result["task"],
            answer=result["answer"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------
# Run: uvicorn webapp.api:app --reload
# ---------------------