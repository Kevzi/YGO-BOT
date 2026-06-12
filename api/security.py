from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

def setup_cors(app: FastAPI):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost",
            "http://localhost:3000",
            "http://localhost:8080",
            "http://127.0.0.1",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
