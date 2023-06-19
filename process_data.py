import json

from transformers import BertTokenizer
from sbert import SBertModel

from config import Config
from functions import load_data, save_data, summarize_abstract


#take a random sample of the data
data = load_data(
    Config.DATA_FILE, 
    Config.CATEGORIZATION_SAMPLE_SIZE, 
    Config.CATEGORIZATION_SAMPLE_SEED)

#Load SBert for extractive summarization
sbert_model = SBertModel.from_pretrained(Config.SBERT_MODEL)
tokenizer = BertTokenizer.from_pretrained(Config.SBERT_MODEL)

#load category definitions
with open("category_defs.json", "rt") as F:
    category_defs = json.load(F)

#function for annotating data
def annotate_record(categories, category_defs):
    for cat in categories.lower().split(" "):
        for cat_def in category_defs:
            for s in category_defs[cat_def]:
                if cat.startswith(s):
                    return cat_def
    return "other"

#annitate target categories and summarize abstracts
for i, record in enumerate(data):
    data[i]["iris_category"] = annotate_record(record["categories"], category_defs)
    data[i]["abstract_summary"] = summarize_abstract(record["abstract"], sbert_model, tokenizer)

#save processed data

save_data(Config.PROCESSED_DATA_FILE, data)
