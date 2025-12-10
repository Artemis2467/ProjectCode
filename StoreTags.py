import json
from TokenRetrieval import return_results

class FileLoader:
    def __init__(self, filepath="general_data.json"):
        self.filepath = filepath
    
    def __iter__(self):
        with open(self.filepath, "r", encoding='utf-8') as f:
            for line in f:
                yield json.loads(line)

def retrieve(dataset, key:str):
    """
    :param key: user_query or chatgpt_response or hallucination or hallucination_spans
    :type key: str
    """
    result = []
    for data in dataset:
        result.append(data[key])
    return result

if __name__ == "__main__":

    halueval = FileLoader()
    queries = retrieve(dataset=halueval, key="user_query")
    response = retrieve(dataset=halueval, key="chatgpt_response")
    tag = retrieve(dataset=halueval, key="hallucination")

    for i in range(len(queries)):

        print(f"{i + 1}/{len(queries)}\n")
        print(f'Query: {queries[i]}\n-------ChatGPT Response-------\n{response[i]}\nHallucination: {tag[i]}')

        for x in range(3):
            results = return_results(prompt=queries[i], max_tokens=100, temperature=0.25, logprobs=5) 
            print(f'\n-------Llama Response-------\n{results["generated_text"]}')
        
            while True:
                try:
                    label = int(input())
                    if label != 1 and label != 0:
                        print("input either 1 or 0")
                    else:
                        break
                except ValueError:
                    print("input either 1 or 0")

            results["label"] = label
            print("\n")
            with open("dataset.jsonl", "a", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)
                f.write("\n")


    

