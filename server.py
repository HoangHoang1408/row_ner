import json
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI

from src.bootstraping import bootstrap
from src.utils.tei_utils import TEIClient

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

embedding_server = os.getenv("EMBEDDING_SERVER")
embedding_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE"))

logger.info(f"Embedding server: {embedding_server}")

tei_client = TEIClient(embedding_server)

with open("resources/database_file.json", "r") as file:
    database = json.load(file)


field_indexes, field_mappings = bootstrap(database, tei_client, embedding_batch_size)


@app.get("/")
def read_root():
    return {"message": "FastAPI server is running."}
