import json
import pandas as pd
from StoreDataset import FileLoader
from AnswerRetrieval import return_results

class LoadDataset():
    def __init__(self, questions, best_ans, cor_ans, inc_ans):
        self.questions = questions
        self.best_ans = best_ans
        self.cor_ans = cor_ans
        self.inc_ans = inc_ans

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        questions = self.questions[idx]
        best_ans = self.best_ans[idx]
        cor_ans = self.cor_ans[idx]
        inc_ans = self.inc_ans[idx]
        return questions, best_ans, cor_ans, inc_ans

def load_data():
    df = pd.read_csv(r'Datasets\TruthfulQA.csv', usecols=["Question", "Best Answer", "Correct Answers", "Incorrect Answers"])
    questions = df["Question"].values
    best_ans = df["Best Answer"].values
    cor_ans = df["Correct Answers"].values
    inc_ans = df["Incorrect Answers"].values

    dataset = LoadDataset(questions, best_ans, cor_ans, inc_ans)
    return dataset

if __name__ == "__main__":

    ans_dataset = FileLoader(r"TruthfullDataset.jsonl")
    dataset = load_data()

    dataset_len = len(dataset)
    ans_len = len(ans_dataset)

    for i in range(dataset_len - ans_len):

        query, best_ans, cor_ans, inc_ans = dataset[ans_len + i - 1]

        print(f"{i + ans_len + 1}/{dataset_len}\n")
        print(f"##-----query-----##\n{query}\n\n-----Best Answer-----\n{best_ans}\n\n-----Correct Answers-----\n{cor_ans}\n\n-----Incorrect Answers-----\n{inc_ans}")

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

            response = result["generated_text"]

            print(f'\n-------Llama Response-------\n{response}\n')
            print(f"\n-----prompt-----\nquery: {query} ans: {response}\n")

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