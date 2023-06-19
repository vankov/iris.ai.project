from sbert import SBertModel
from transformers import BertTokenizer
from fastapi import FastAPI, Body
from config import Config
from functions import summarize_abstract

app = FastAPI()
model = SBertModel.from_pretrained(Config.MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(Config.SBERT_MODEL)

@app.post("/categorize_abstract")
async def categorize_abstract(abstract: str = Body()):
    """
        Abstract categorization endpoint
    """

    #summarize abstract
    summary = summarize_abstract(abstract, model, tokenizer)
    #predicate category
    label, score = model.predict_from_text(summary, tokenizer)

    return {
        "category": label,
        "score": str(score)
    }
