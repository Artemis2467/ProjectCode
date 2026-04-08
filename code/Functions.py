import os
import math
import re
import json
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, f1_score, classification_report, precision_recall_curve
from sentence_transformers import SentenceTransformer
from StoreDataset import FileLoader, retrieve_to

class LinearConfig:

    pos_weight = torch.tensor(1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_p = 0.75
    val_p = 0.15

    d_model_choices = [32, 64, 128, 256]

    num_epochs = 50
    batch_num = 32
    learning_rate_choices = [0.1, 0.05, 0.01, 0.005, 0.001]
    stop_patience = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learning_rate = 0.1
    d_model = 32

class LogitConfig:

    pos_weight = torch.tensor(0.7)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    add_conv_choices = [False, True]

    train_p = 0.75
    val_p = 0.15

    d_model_choices = [32, 64, 128, 256]
    conv_ch_choices = [32, 64]
    drop_out = 0.1

    num_epochs = 100
    batch_num = 32
    learning_rate_choices = [0.1, 0.05, 0.01, 0.005, 0.001]
    patience = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    add_conv = False
    d_model = 64
    conv_ch = 64
    learning_rate = 0.05

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
              epoch:int | None=None
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

                output = torch.sigmoid(output.squeeze(1))
                results.append(output.tolist())
                targets.append(batch["labels"].tolist())
        
        results = [item for res in results for item in res]
        targets = [item for tar in targets for item in tar]
        
        return results, targets

    else:
        raise RuntimeError("type is not recognized")

    return total_loss


# ————————————————————————————————
# plot graphs and caculate scores
# ————————————————————————————————

def plot_loss(history, is_linear, show, length=None):
    plt.figure(figsize=(8, 5))
    plt.plot(history["running_loss"], label='running_loss', color='blue')
    plt.plot(history["val_loss"], label='val_loss', color='red', linestyle="dashed")
    plt.title('validation vs running loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()

    if is_linear:
        if not length:
            length = count_files(r"test_results\loss\linear") + 1
        plt.savefig(fr"test_results\loss\linear\linear_{length}.pdf")
    else:
        if not length:
            length = count_files(r"test_results\loss\logprob") + 1
        plt.savefig(fr"test_results\loss\logprob\logprob_{length}.pdf")

    if show:
        plt.show()

def calculate_threshold(model, config, val_loader):

    model.eval()
    results = []
    targets = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(config.device) for k, v in batch.items()}

            if model.is_linear:
                y_pred = model(batch["cos_sim"], batch["entropy"])
            else:
                y_pred = model(
                    batch["logprobs1"],
                    batch["logprobs2"],
                    batch["cos_sim"],
                    batch["entropy"]
                )
            y_pred = torch.sigmoid(y_pred.squeeze(1))
            results.append(y_pred.tolist())
            targets.append(batch["labels"].tolist())

    results = [item for res in results for item in res]
    targets = [item for tar in targets for item in tar]

    precision, recall, thresholds = precision_recall_curve(targets, results)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    f1_scores = f1_scores[:-1]
    best_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_index]

    return best_threshold

def test_model(config, test_loader, threshold, model, parameter_path, plot_distribution:bool = False):
    model.load_state_dict(torch.load(os.path.join("models", parameter_path)))

    y_pred, y_true = run_batch(
        config,
        test_loader,
        model,
        None,
        "test",
    )

    if plot_distribution:
        plt.close(1)
        plt.figure(2)
        sns.histplot(data=y_pred, bins=5, kde=True, color="teal", label="model output")
        plt.xlim(0, 1)

        if input("Print 0.5 threshold? [y/n] ") == "y":
            plt.axvline(0.5, color="b", linestyle="dashed", label="0.5 threshold")
        if input("Print calculated threshold? [y/n] ") == "y":
            plt.axvline(threshold, color="r", linestyle="dashed", label="calculated threshold")
        plt.title("distribution of model output")
        plt.legend()

        save = input("save? [y/n] ") == "y"

        if save:
            name = input("input name: ")
            plt.savefig(fr"test_results\{name}.png")

        plt.show()

    fpr, tpr, thre = roc_curve(y_true, y_pred)
    auroc = auc(fpr, tpr)
    y_pred = [0 if result < threshold else 1 for result in y_pred] 
    f1 = f1_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return auroc, f1, fpr, tpr, report, report_dict


def graph_roc_curve(auroc, fpr, tpr, is_linear, length=None):

    plt.figure()  
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auroc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {"linear model" if is_linear else "logprob model"}')
    plt.legend()

    if is_linear:
        if not length:
            length = count_files(r"test_results\ROC\linear") + 1
        plt.savefig(fr"test_results\ROC\linear\{length}.pdf")
    else:
        if not length:
            length = count_files(r"test_results\ROC\logprob") + 1
        plt.savefig(fr"test_results\ROC\logprob\{length}.pdf")

if __name__ == "__main__":
    store_data(ans_dataset="TruthfulDataset.jsonl", cal_dataset="CalDataset.jsonl")