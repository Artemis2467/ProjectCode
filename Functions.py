from sentence_transformers import SentenceTransformer
from StoreDataset import FileLoader

def retrieve_logprobs():
    dataset = FileLoader("dataset.jsonl")
    for obj in dataset:
        result = []
        for i in range(2):
            result.append(obj[f"{i}"]["top_logprobs"])
        yield result

def retrieve_label():
    dataset = FileLoader("dataset.jsonl")
    for obj in dataset:
        result = []
        for i in range(2):
            result.append(obj[f"{i}"]["label"])
        yield result

def retrieve_text():
    dataset = FileLoader("dataset.jsonl")
    for obj in dataset:
        result = []
        for i in range(2):
            result.append(obj[f"{i}"]["generated_text"])
        yield result

def retrieve_embed():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    for obj in retrieve_text():
        embeddings = []
        for text in obj:
            embed = model.encode(text[0])
            embeddings.append(embed)
        yield embeddings
