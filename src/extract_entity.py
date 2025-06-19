import logging
import os
from collections import defaultdict

from dotenv import load_dotenv
from transformers import AutoModelForTokenClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
if not MODEL_PATH:
    logger.error("MODEL_PATH not set in .env file")
    raise ValueError("MODEL_PATH not set in .env file")

logger.info(f"Loading tokenizer from {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
logger.info(f"Loading model from {MODEL_PATH}")
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH).to("cpu")
model.eval()


def extract_entity(text):
    def convert_tokens_to_string(tokens):
        text = ""
        for token in tokens:
            if "▁" in token:
                text = text + " " + token.replace("▁", "")
            else:
                text = text + token
        return text.strip()

    def tag_text(text, model, tokenizer, id2label):
        inputs = tokenizer(
            text.split(), truncation=True, is_split_into_words=True, return_tensors="pt"
        ).to(model.device)
        outputs = model(**inputs)
        preds = outputs.logits.argmax(2)[0].tolist()
        words = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        labels = [id2label[pred] for pred in preds]
        return words, labels, inputs.input_ids[0]

    words, labels, ids = tag_text(text, model, tokenizer, model.config.id2label)
    info = defaultdict(list)
    group = []
    for word, label, idx in zip(words, labels, ids):
        if idx == tokenizer.cls_token_id or idx == tokenizer.sep_token_id:
            continue
        if word.startswith("▁"):
            group.append({"tokens": [], "labels": []})
        group[-1]["tokens"].append(word)
        group[-1]["labels"].append(label)
        for g in group:
            g["main_label"] = g["labels"][0]

    new_group = []
    for i, g in enumerate(group):
        if i == 0:
            new_group.append(g)
        else:
            if (
                g["main_label"].startswith("I-")
                and g["main_label"][2:] == new_group[-1]["main_label"][2:]
            ):
                new_group[-1]["tokens"] += g["tokens"]
                new_group[-1]["labels"] += g["labels"]
            else:
                new_group.append(g)

    for g in new_group:
        key = g["main_label"][2:]
        if key:
            value = convert_tokens_to_string(g["tokens"])
            info[key].append(value)

    return dict(info)
