import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np

class LoadDataset(Dataset):
    def __init__(self, questions, statements, labels):
        self.questions = questions
        self.statements = statements
        self.labels = labels

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        questions = self.questions[idx]
        statements = self.statements[idx]
        label = self.labels[idx]
        return questions, statements, label

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def LoadData():
    df = pd.read_csv('TruthfulQA.csv', usecols=["Question", "Correct Answers", "Incorrect Answers"])
    questions = np.tile(df["Question"].values, reps=2)
    statements = [x.split("; ") for x in df["Correct Answers"].values] + [x.split("; ") for x in df["Incorrect Answers"].values]
    label = [1 if i <= len(questions) // 2 else 0 for i in range(len(questions))]

    # The dataset is set in this order (Question, statements, label)
    # the label indicates factuality (True or False)
    dataset = LoadDataset(questions, statements, label)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    return dataset, loader


#If you want to train model specifically based on dataset
# (which is relatively too small) use this function to generate corpus
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

if __name__ == "__main__":
    dataset, loader = LoadData()
    corp = LoadCorpus(dataset)
    print(corp)
    print(dataset)
    print(loader)