from datasets import load_dataset
from Functions import LinearConfig, LogitConfig
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split
    
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

logitconfig = LogitConfig()
linearconfig = LinearConfig()

raw_dataset =  load_dataset("json", data_files=r"Datasets\TruthfulDataset.jsonl", split="train")
raw_dataset_cal = load_dataset("json", data_files=r"Datasets\CalDataset.jsonl", split="train")

# Set logprob dataset
length = len(raw_dataset)

train_len = int(logitconfig.train_p * length)
val_len = int(logitconfig.val_p * length)
test_len = length - train_len - val_len

logprob_dataset = LogprobDataset(
    lp_dataset=raw_dataset,
    cal_dataset=raw_dataset_cal
)

logprob_train, logprob_val, logprob_test = random_split(
    logprob_dataset,
    [train_len, val_len, test_len],
    generator=torch.Generator().manual_seed(42)
)

logprob_train_loader = DataLoader(
    dataset=logprob_train, 
    batch_size=logitconfig.batch_num,
    shuffle=True,
    collate_fn=collate_fn, # using a defined collate function, data length is different for every input
)

logprob_val_loader = DataLoader(
    dataset=logprob_val, 
    batch_size=logitconfig.batch_num,
    shuffle=True,
    collate_fn=collate_fn, 
)

logprob_test_loader = DataLoader(
    dataset=logprob_test, 
    batch_size=logitconfig.batch_num,
    shuffle=True,
    collate_fn=collate_fn, 
)

# Set linear dataset
linear_dataset = LinearDataset(
    orig_dataset=raw_dataset,
    cal_dataset=raw_dataset_cal
)

length = len(linear_dataset)

train_len = int(linearconfig.train_p * length)
val_len = int(linearconfig.val_p * length)
test_len = length - train_len - val_len

linear_train, linear_val, linear_test = random_split(
    linear_dataset,
    [train_len, val_len, test_len],
    generator=torch.Generator().manual_seed(42),
)

linear_train_loader = DataLoader(
    dataset=linear_train,
    batch_size=linearconfig.batch_num,
    shuffle=True,
)

linear_val_loader = DataLoader(
    dataset=linear_val,
    batch_size=linearconfig.batch_num,
    shuffle=True,
)

linear_test_loader = DataLoader(
    dataset=linear_test,
    batch_size=linearconfig.batch_num,
    shuffle=True,
)