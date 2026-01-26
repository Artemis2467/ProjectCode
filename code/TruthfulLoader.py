import pandas as pd
import numpy as np
from torch.utils.data import Dataset

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
    df = pd.read_csv(r'Datasets\TruthfulQA.csv', usecols=["Question", "Correct Answers", "Incorrect Answers"])
    questions = np.tile(df["Question"].values, reps=2)
    statements = [x.split("; ") for x in df["Correct Answers"].values] + [x.split("; ") for x in df["Incorrect Answers"].values]
    label = [1 if i <= len(questions) // 2 else 0 for i in range(len(questions))]

    # The dataset is set in this order (Question, statements, label)
    # the label indicates factuality (True or False)
    dataset = LoadDataset(questions, statements, label)
    return dataset

if __name__ == "__main__":
    dataset= LoadData()
    print(dataset)