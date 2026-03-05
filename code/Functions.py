import os
import math
import re
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from sentence_transformers import SentenceTransformer
from StoreDataset import FileLoader, retrieve_to

class LinearConfig:

    criterion = nn.BCELoss()

    train_p = 0.75
    val_p = 0.125

    d_model = 32

    num_epochs = 50
    batch_num = 32
    learning_rate = 0.01
    stop_patience = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LogitConfig:

    criterion = nn.BCELoss()

    train_p = 0.75
    val_p = 0.125

    d_model = 32
    drop_out = 0.2

    num_epochs = 50
    batch_num = 32
    learning_rate = 0.01
    patience = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_files(dir):
    files = [f for f in os.listdir(dir)]
    return len(files)

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

def run_batch(config, 
              loader, 
              model, 
              optimizer,
              type: str, 
              epoch: int | None
              ):
    
    total_loss = 0.0

    if type == "train":
        model.train()
    
        for batch in tqdm(loader, desc=f"epoch: {config.num_epochs}/{epoch + 1}"):
            batch = {k: v.to(config.device) for k, v in batch.items()}

            optimizer.zero_grad()
            if model.is_linear:
                output = model(batch["cos_sim"], batch["entropy"])
            else:
                output = model(
                    batch["logprobs1"],
                    batch["logprobs2"],
                    batch["cos_sim"],
                    batch["entropy"]
                )

            loss = config.criterion(output, batch["labels"].unsqueeze(1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch["cos_sim"].size(0)

    elif type == "val":
        model.eval()

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(config.device) for k, v in batch.items()}

                if model.is_linear:
                    output = model(batch["cos_sim"], batch["entropy"])
                else:
                    output = model(
                        batch["logprobs1"],
                        batch["logprobs2"],
                        batch["cos_sim"],
                        batch["entropy"]
                    )

                loss = config.criterion(output, batch["labels"].unsqueeze(1))

                total_loss += loss.item() * batch["cos_sim"].size(0)
    
    elif type == "test":
        model.eval()
        results = []
        targets = []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(config.device) for k, v in batch.items()}

                if model.is_linear:
                    output = model(batch["cos_sim"], batch["entropy"])
                else:
                    output = model(
                        batch["logprobs1"],
                        batch["logprobs2"],
                        batch["cos_sim"],
                        batch["entropy"]
                    )
                results.append(output.squeeze(1).tolist())
                targets.append(batch["labels"].tolist())
        
        results = [item for res in results for item in res]
        targets = [item for tar in targets for item in tar]
        
        return results, targets

    else:
        raise RuntimeError("type is not recognized")

    
    return total_loss

def plot_loss(history, is_linear):
    plt.figure(figsize=(8, 5))
    plt.plot(history["running_loss"], label='running_loss', color='blue')
    plt.plot(history["val_loss"], label='val_loss', color='red', linestyle="dashed")
    plt.title('validation vs running loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()

    length = count_files(r"test_results\loss")

    if is_linear:
        plt.savefig(os.path.join(r"test_results\loss", f"linear_{length}.pdf"))
    else:
        plt.savefig(os.path.join(r"test_results\loss", f"logprob_{length}.pdf"))
    plt.show()

def graph_roc_curve(config, test_loader, model, parameter_path: str):

    model.load_state_dict(torch.load(os.path.join("models", parameter_path)))

    results, targets = run_batch(
        config,
        test_loader,
        model,
        None,
        "test",
        None
    )

    fpr, tpr, threshold = roc_curve(targets, results)
    auroc = auc(fpr, tpr)

    plt.figure()  
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auroc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {"linear model" if model.is_linear else "logprob model"}')
    plt.legend()

    length = count_files(r"test_results\ROC")
    if model.linear:
        plt.savefig(os.path.join(r"test_results\ROC", f"linear_{length // 2}.pdf"))
    else:
        plt.savefig(os.path.join(r"test_results\ROC", f"logprob_{length // 2}.pdf"))

    plt.show()

if __name__ == "__main__":
    store_data(ans_dataset="TruthfulDataset.jsonl", cal_dataset="CalDataset.jsonl")