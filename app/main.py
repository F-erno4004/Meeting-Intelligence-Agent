
from fastapi import FastAPI

app = FastAPI(
    title="Meeting Intelligence Agent",
    description="An AI agent for meeting memory and intelligence",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Meeting Intelligence Agent is running!"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }
