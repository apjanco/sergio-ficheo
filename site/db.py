import srsly
import typer
import chromadb
from pathlib import Path
from chromadb.utils import embedding_functions

def make_index(db_name:str, data:list):
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")

    chroma_client = chromadb.PersistentClient(path=db_name)
    collection = chroma_client.get_or_create_collection(name=db_name, embedding_function=sentence_transformer_ef)

    documents = [d['text'] for d in data]
    embeddings = sentence_transformer_ef(documents)
    ids = [d['image'] for d in data]
    collection.add(
        embeddings=embeddings,
        documents=documents,
        ids=ids
    )

if __name__ == "__main__":
    data = list(srsly.read_jsonl("../data/data.jsonl"))
    # sort the data by image name
    data = sorted(data, key=lambda x: x['image'])

    db_name = "sergioDB"
    if not Path(db_name).exists():
        print("[teal]Creating index...[/teal]")
        make_index(db_name, data)

