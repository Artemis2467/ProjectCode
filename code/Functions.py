import os
import math
import re
import json
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from StoreDataset import FileLoader, retrieve_to

def retrieve_logprobs():
    dataset = FileLoader("AnsDataset.jsonl")
    for obj in dataset:
        result = []
        result.append(obj["0"]["top_logprobs"])
        result.append(obj["1"]["top_logprobs"])
        yield result

def retrieve_label():
    dataset = FileLoader("AnsDataset.jsonl")
    for obj in dataset:
        result = []
        result.append(obj["0"]["label"])
        result.append(obj["1"]["label"])
        yield result

def retrieve_tag():
    halueval = FileLoader()
    tag = retrieve_to(dataset=halueval, key="hallucination")
    return tag

def retrieve_text():
    dataset = FileLoader("AnsDataset.jsonl")
    for obj in dataset:
        result = []
        result.append(obj["0"]["generated_text"])
        result.append(obj["1"]["generated_text"])
        yield result

def retrieve_embed():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    for obj in retrieve_text():
        embeddings = []
        for text in obj:
            cleaned_text = re.sub("[|*-`]", " ", text)
            embed = model.encode(cleaned_text)
            embeddings.append(embed)
        yield embeddings

def avg_perplexity(logprobs):
    count = 0
    total = 0
    for token in logprobs:
        for token_prob in token.values():
            if token_prob is None:
                continue
            total += token_prob * math.exp(token_prob)
        count += 1
    result =  math.exp(total / count * -1)
    return result

def store_data():

    for i, (embed, logprobs) in enumerate(zip(retrieve_embed(), retrieve_logprobs())):

        first_embedding = torch.from_numpy(embed[0])
        second_embedding = torch.from_numpy(embed[1])
        text_perplexity = 0

        for logprob in logprobs:
            text_perplexity += avg_perplexity(logprob)

        avg_p = text_perplexity / 2
        cos_sim = F.cosine_similarity(first_embedding, second_embedding, dim=0).tolist()
        res = {i: (cos_sim, avg_p)}

        with open(os.path.join("Datasets", "CalDataset.jsonl"), "a", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False)
            f.write("\n")
