import os
import math
import re
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from StoreDataset import FileLoader, retrieve_to

def LoadCorpus(dataset):
    corpus = {}
    idx = 0
    for i in range(len(dataset)):
        for word in dataset[i][0][:-1].split():
            if word.lower() not in corpus.values():
                corpus[idx] = word.lower()
                idx += 1
        for x in range(len(dataset[i][1])):
            for word in dataset[i][1][x].split():
                if word.lower() not in corpus.values():
                    corpus[idx] = word.lower()
                    idx += 1
    return corpus

def retrieve_tag():
    halueval = FileLoader()
    tag = retrieve_to(dataset=halueval, key="hallucination")
    return tag

def retrieve_logprobs(ans_dataset: str):
    dataset = FileLoader(ans_dataset)
    for obj in dataset:
        result = []
        result.append(obj["top_logprobs1"])
        result.append(obj["top_logprobs2"])
        yield result

def retrieve_text(ans_dataset: str):
    dataset = FileLoader(ans_dataset)
    for obj in dataset:
        result = []
        result.append(obj["text1"])
        result.append(obj["text2"])
        yield result

def retrieve_embed(ans_dataset: str):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    for obj in retrieve_text(ans_dataset):
        embeddings = []
        for text in obj:
            cleaned_text = re.sub("[|*-`]", " ", text)
            embed = model.encode(cleaned_text)
            embeddings.append(embed)
        yield embeddings

def logit_ent(logprobs):
    text_len = len(logprobs)
    total = 0.0
    for token in logprobs:
        for token_prob in token:
            if token_prob == -1e2:
                continue
            total += token_prob * math.exp(token_prob)
    result =  -1 * (total / text_len)
    return result

def store_data(ans_dataset: str, cal_dataset: str):

    for i, (embed, logprobs) in tqdm(enumerate(zip(retrieve_embed(ans_dataset), retrieve_logprobs(ans_dataset)))):

        embedding1 = torch.from_numpy(embed[0])
        embedding2 = torch.from_numpy(embed[1])

        ent1 = logit_ent(logprobs[0])
        ent2 = logit_ent(logprobs[1])

        cos_sim = F.cosine_similarity(embedding1, embedding2, dim=0).tolist()

        res1 = {"cos_sim": cos_sim, "entropy": ent1}
        res2 = {"cos_sim": cos_sim, "entropy": ent2}

        with open(os.path.join("Datasets", cal_dataset), "a", encoding="utf-8") as f:
            json.dump(res1, f, ensure_ascii=False)
            f.write("\n")
            json.dump(res2, f, ensure_ascii=False)
            f.write("\n")
