import json
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from database import initialize_bootstrap_resources
from src.extract_entity import extract_entity
from src.utils.tei_utils import TEIClient

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Move all bootstrapping logic to bootstraping_indices.py
tei_client, database, field_indexes, field_mappings = initialize_bootstrap_resources()

embedding_server = os.getenv("EMBEDDING_SERVER")
embedding_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE"))

logger.info(f"Embedding server: {embedding_server}")

logger.info(f"Field indexes: {field_indexes}")
logger.info(f"Field mappings: {field_mappings}")


class Query(BaseModel):
    text: str


@app.get("/")
def read_root():
    return {"message": "FastAPI server is running."}


@app.post("/entity_extractor")
async def entity_extractor_endpoint(query: Query):
    if len(query.text) == 0:
        return {}
    return extract_entity(query.text)


@app.post("/search_rows")
async def search_rows_endpoint(query: Query):
    entity_dict = extract_entity(query.text)
    
