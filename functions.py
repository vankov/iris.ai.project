import json
import numpy as np
import tensorflow as tf
from nltk.tokenize import sent_tokenize
from lexrank import degree_centrality_scores
from config import Config

def load_data(data_file, sample_size = None, sample_seed = None):
    """
        Load data from a JSON file and generate a random sample from it
    """
    data = []
    with open(data_file, "rt") as F:
        for line in F.readlines():
            record = json.loads(line)
            data.append(record)        

    if sample_seed is not None:
        np.random.seed(sample_seed)
    np.random.shuffle(data)

    if sample_size is None:
        return data
    else:
        return data[:sample_size]      
    
def save_data(data_file, data):
    """
        Save data in a JSON file
    """    
    with open(data_file, "wt") as F:
        F.writelines([json.dumps(r) + "\n" for r in data])
        
def cosine_similarity(tensor1, tensor2):
    """
        Compute cosine similarity of two tensors. The tensors can have 1 or 2 dimensions
    """        
    if not tf.is_tensor(tensor1):
        tensor1 = tf.constant(tensor1, dtype=tf.float32)
    if not tf.is_tensor(tensor2):
        tensor2 = tf.constant(tensor2, dtype=tf.float32)
    if len(tensor1.shape) == 1:
        tensor1 = tf.expand_dims(tensor1, 0)
    if len(tensor2.shape) == 1:
        tensor2 = tf.expand_dims(tensor2, 0)
    tensor1_norm = tf.nn.l2_normalize(tensor1, 1)
    tensor2_norm = tf.nn.l2_normalize(tensor2, 1)

    return tf.matmul(tensor1_norm, tensor2_norm, transpose_b=True)

def summarize_abstract(abstract, sbert_model, tokenizer):
    """
        Summarizes an abstract(i.e. text) using the LexRank algorithm
    """            
    sentences = sent_tokenize(abstract)
    embeddings = sbert_model.encode(sentences, tokenizer)
    similarity_matrix = cosine_similarity(embeddings, embeddings).numpy()
    centrality_scores = degree_centrality_scores(similarity_matrix, threshold=None)

    return sentences[np.argmax(centrality_scores)]