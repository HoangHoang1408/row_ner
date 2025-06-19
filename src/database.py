import json
import logging
import os
import pickle
from typing import Dict, List

import faiss
import numpy as np
from dotenv import load_dotenv

from src.utils.tei_utils import TEIClient

logger = logging.getLogger(__name__)


def embed_database(database: List[Dict], tei_client, embedding_batch_size):
    fields_to_index = ["name", "email"]
    field_indexes = {}
    field_mappings = {}
    for field in fields_to_index:
        texts = []
        idx_map = []
        for i, entry in enumerate(database):
            value = entry.get(field)
            if value:
                texts.append(value)
                idx_map.append(i)
        if not texts:
            continue
        embeddings = []
        for i in range(0, len(texts), embedding_batch_size):
            batch = texts[i : i + embedding_batch_size]
            batch_embeddings = tei_client.embed(batch)
            embeddings.extend(batch_embeddings)
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        n_embeddings = embeddings.shape[0]
        nlist = max(10, int(np.sqrt(n_embeddings)))
        if n_embeddings > nlist:
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(
                quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
            )
            index.train(embeddings)
            index.add(embeddings)
        else:
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
        field_indexes[field] = index
        field_mappings[field] = {i: database[idx_map[i]] for i in range(len(idx_map))}
        logger.info(f"FAISS index for field '{field}': {index}")
        logger.info(
            f"Index to data mapping for field '{field}': {field_mappings[field]}"
        )
    return field_indexes, field_mappings


def save_faiss_index(index, path):
    faiss.write_index(index, path)


def load_faiss_index(path):
    return faiss.read_index(path)


def save_mappings(mappings, path):
    with open(path, "wb") as f:
        pickle.dump(mappings, f)


def load_mappings(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def bootstrap_indices(
    database, tei_client, embedding_batch_size, index_dir="resources/faiss_indices"
):
    os.makedirs(index_dir, exist_ok=True)
    fields_to_index = ["name", "email"]
    field_indexes = {}
    field_mappings = {}
    # Try to load indices and mappings if they exist
    all_exist = True
    for field in fields_to_index:
        index_path = os.path.join(index_dir, f"{field}.index")
        mapping_path = os.path.join(index_dir, f"{field}_mapping.pkl")
        if not (os.path.exists(index_path) and os.path.exists(mapping_path)):
            all_exist = False
            break
    if all_exist:
        for field in fields_to_index:
            index_path = os.path.join(index_dir, f"{field}.index")
            mapping_path = os.path.join(index_dir, f"{field}_mapping.pkl")
            field_indexes[field] = load_faiss_index(index_path)
            field_mappings[field] = load_mappings(mapping_path)
        logger.info("Loaded FAISS indices and mappings from disk.")
        return field_indexes, field_mappings
    # Otherwise, create them
    field_indexes, field_mappings = embed_database(
        database, tei_client, embedding_batch_size
    )
    for field in fields_to_index:
        index_path = os.path.join(index_dir, f"{field}.index")
        mapping_path = os.path.join(index_dir, f"{field}_mapping.pkl")
        save_faiss_index(field_indexes[field], index_path)
        save_mappings(field_mappings[field], mapping_path)
    logger.info("Created and saved FAISS indices and mappings.")
    return field_indexes, field_mappings


def load_indices():
    load_dotenv()
    embedding_server = os.getenv("EMBEDDING_SERVER")
    embedding_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE"))
    logger.info(f"Loading TEIClient with server: {embedding_server}")
    tei_client = TEIClient(embedding_server)
    logger.info("Loading database from resources/database_file.json")
    with open("resources/database_file.json", "r") as file:
        database = json.load(file)
    logger.info("Bootstrapping or loading FAISS indices and mappings...")
    field_indexes, field_mappings = bootstrap_indices(
        database, tei_client, embedding_batch_size
    )
    logger.info("Indices and mappings are ready.")
    return tei_client, database, field_indexes, field_mappings


tei_client, database, field_indexes, field_mappings = load_indices()


def find_row(query_value, query_field, top_k=1):
    index = field_indexes[query_field]
    query_embedding = tei_client.embed([query_value])
    _, indices = index.search(query_embedding, top_k)
    return [field_mappings[query_field][i] for i in indices[0]]
