from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from text_converter import convert_text

app = FastAPI()
templates = Jinja2Templates(directory="src")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class TextRequest(BaseModel):
    text: str
    style: str

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/transform")
async def transform(request: Request):
    return templates.TemplateResponse("transform.html", {"request": request})

@app.post("/convert")
async def convert(request: TextRequest):
    converted_text = convert_text(request.text, request.style)
    return {"converted_text": converted_text}

@app.get("/styles")
async def get_styles():
    return {
        "styles": [
            "공손체(존댓말)",
            "귀여워",
            "반말체",
            "??"
        ]
    }

app.mount("/src", StaticFiles(directory="src"), name="src")