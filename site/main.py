import os
import secrets
import srsly
import markdown
from typing import Annotated
from rich import print
from db import make_index
from pathlib import Path
from fastapi import FastAPI, Request, Depends, HTTPException, status, Form
from fastapi.security import OAuth2PasswordBearer
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import chromadb
from chromadb.utils import embedding_functions


data = list(srsly.read_jsonl("../data/data.jsonl"))
# sort the data by image name
data = sorted(data, key=lambda x: x['image'])

app = FastAPI()
security = HTTPBasic()
app.mount("/assets", StaticFiles(directory='../splits'), name="assets")
templates = Jinja2Templates(directory="templates")

## set up the Chroma database
db_name = "sergioDB"
if not Path(db_name).exists():
    print("[teal]Creating index...[/teal]")
    make_index(db_name, data)

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
chroma_client = chromadb.PersistentClient(path=db_name)
collection = chroma_client.get_or_create_collection(name=db_name, embedding_function=sentence_transformer_ef)

def get_current_username(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
):
    current_username_bytes = credentials.username.encode("utf8")
    correct_username_bytes = os.getenv("USERNAME", "admin").encode("utf8")
    is_correct_username = secrets.compare_digest(
        current_username_bytes, correct_username_bytes
    )
    current_password_bytes = credentials.password.encode("utf8")
    correct_password_bytes = os.getenv("PASSWORD", "admin").encode("utf8")
    is_correct_password = secrets.compare_digest(
        current_password_bytes, correct_password_bytes
    )
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@app.get("/")
def read_root(request: Request, username: Annotated[str, Depends(get_current_username)]):
    return templates.TemplateResponse("index.html", {"request": request, "username": username})

@app.post("/")
async def search_item(request: Request):
    post_data = await request.json()
    results = collection.query(
        query_texts = [post_data["search"]],
        n_results=1000,
        where_document={"$contains":post_data["search"]}
    )
    sorted_results = []
    #TODO read text from jsonl file rather than index, changes can cause confusion
    for id, document, distance in zip(results['ids'][0], results['documents'][0], results['distances'][0]):
        sorted_results.append({"image":id, "text":document, "distance":distance})
    return sorted_results

@app.get("/page/{image}")
def read_item(image: str , request: Request):
    item = next((item for item in data if item["image"] == image), None)
    if item:
        item_idx = data.index(item)
        previous = data[item_idx - 1]["image"] if item_idx > 0 else None
        item['previous'] = f"/page/{previous}" if previous else None
        next_ = data[item_idx + 1]["image"] if item_idx < len(data) - 1 else None
        item['next'] = f"/page/{next_}" if next_ else None
        item['text'] = markdown.markdown(item['text'])
        return templates.TemplateResponse(
                        "page.html",
                        {"request": request, "item": item},
                    )
    else:
        return {"image": image, "error": "Image not found"}

@app.post("/page/{image}")
async def write_item(user: Annotated[str, Depends(get_current_username)], image:str, request: Request):
    post_data = await request.json()
    item = next((item for item in data if item["image"] == image), None)
    item['text'] = post_data['text']
    #write updated data to jsonl file
    srsly.write_jsonl("data/data.jsonl", data)
    
    