from typing import Optional
from fastapi import FastAPI


app = FastAPI()


@app.get("/")
def get_root():
    return {"title": "Wow That's Fast Labeling"}
