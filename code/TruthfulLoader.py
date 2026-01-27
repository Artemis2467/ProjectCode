import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from StoreDataset import FileLoader
from AnswerRetrieval import return_results

class LoadDataset(Dataset):
    def __init__(self, questions, statements):
        self.questions = questions
        self.statements = statements

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        questions = self.questions[idx]
        statements = self.statements[idx]
        return questions, statements

def load_data():
    df = pd.read_csv(r'Datasets\TruthfulQA.csv', usecols=["Question", "Best Answer",])
    questions = np.array(df["Question"].values)
    ans = np.array(df["Best Answer"])

    dataset = LoadDataset(questions, ans)
    return dataset

if __name__ == "__main__":

    ans_dataset = FileLoader(r"TruthfullDataset.jsonl")
    dataset = load_data()

    dataset_len = len(dataset)
    ans_len = len(ans_dataset)

    for i in range(dataset_len - ans_len):

        query, ans = dataset[ans_len + i]

        print(f"{i + ans_len + 1}/{dataset_len}\n")
        print(f"##-----query-----##\n{query}\n\n-----Best Answer-----\n{ans}\n")

        results = {}
        results["label"] = []

        for x in range(2):

            result = return_results(
                model="meta-llama/Llama-3.2-3B-Instruct-Turbo",
                prompt=query,
                temperature=1.1,
                logprobs=5,
                top_k=10,
                top_p=0.9,
                max_tokens=75,
            )

            print(f'\n-------Llama Response-------\n{result["generated_text"]}\n')
            while True:
                try:
                    label = int(input())
                    if label != 1 and label != 0:
                        print("input either 1 or 0")
                    else:
                        break
                except ValueError:
                    print("input either 1 or 0")

            results["label"].append(label)
            results[f"text{x + 1}"] = result["generated_text"]
            results[f"top_logprobs{x + 1}"] = result["top_logprobs"]

        with open(r"Datasets\TruthfullDataset.jsonl", "a", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)
            f.write("\n")
        print("---Successfully stored---\n")

            