from datasets import load_dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

raw_dataset =  load_dataset("json", data_files=r"Datasets\AnsDataset.jsonl", split="train")
raw_dataset_cal = load_dataset("json", data_files=r"Datasets\CalDataset.jsonl", split="train")
    
class LogprobDataset(Dataset):
    def __init__(self, lp_dataset, cal_dataset):
        super().__init__()
        
        self.lp = lp_dataset
        self.cal = cal_dataset

    def __len__(self):

        return len(self.lp)
    
    def __getitem__(self, idx):

        label = 0
        if self.lp["label"][idx][0] or self.lp["label"][idx][1]:
            label = 1
        
        label = torch.tensor(label).float()
        logprobs1 = torch.tensor(self.lp["top_logprobs1"][idx]).float()
        logprobs2 = torch.tensor(self.lp["top_logprobs2"][idx]).float()

        cos_sim = torch.tensor(self.cal["cos_sim"][idx * 2]).float()
        entropy1 = self.cal["entropy"][idx * 2]
        entropy2 = self.cal["entropy"][idx * 2 + 1]

        mean_entropy = torch.tensor((entropy1 + entropy2) / 2).float()

        return {
            "logprobs": (logprobs1, logprobs2), 
            "entropy": mean_entropy, 
            "cos_sim": cos_sim, 
            "label": label
        }
    
class LinearDataset(Dataset):
    def __init__(self, orig_dataset, cal_dataset):
        super().__init__()

        self.org = orig_dataset
        self.cal = cal_dataset

    def __len__(self):

        return len(self.cal)
    
    def __getitem__(self, idx):

        label = torch.tensor(self.org["label"][idx // 2][idx % 2]).float()

        cos_sim = torch.tensor(self.cal["cos_sim"][idx]).float()
        entropy = torch.tensor(self.cal["entropy"][idx]).float()

        return {
            "cos_sim": cos_sim, 
            "entropy": entropy, 
            "label": label
        }
    
def collate_fn(batch):
    logprobs1 = [data["logprobs"][0] for data in batch]
    logprobs2 = [data["logprobs"][1] for data in batch]
    cos_sim = [data["cos_sim"] for data in batch]
    entropy = [data["entropy"] for data in batch]
    labels = [data["label"] for data in batch]

    logprobs1 = pad_sequence(logprobs1, batch_first=True, padding_value=-100.0)
    logprobs2 = pad_sequence(logprobs2, batch_first=True, padding_value=-100.0)
    cos_sim = torch.stack(cos_sim)
    entropy = torch.stack(entropy)
    labels = torch.stack(labels)

    return {
        "logprobs1": logprobs1,
        "logprobs2": logprobs2,
        "cos_sim": cos_sim,
        "entropy": entropy,
        "labels": labels
    }

logprob_dataset = LogprobDataset(
    lp_dataset=raw_dataset,
    cal_dataset=raw_dataset_cal
)

logprob_loader = DataLoader(
    dataset=logprob_dataset, 
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn,
)

linear_dataset = LinearDataset(
    orig_dataset=raw_dataset,
    cal_dataset=raw_dataset_cal
)

linear_loader = DataLoader(
    dataset=linear_dataset,
    batch_size=32,
    shuffle=True,
)