import json
from AnswerRetrieval import return_results

class FileLoader:
    def __init__(self, filepath="general_data.json"):
        self.filepath = filepath
    
    def __iter__(self):
        with open(self.filepath, "r", encoding='utf-8') as f:
            for line in f:
                yield json.loads(line)

def retrieve_to(dataset, key:str):
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
    queries = retrieve_to(dataset=halueval, key="user_query")
    response = retrieve_to(dataset=halueval, key="chatgpt_response")
    tag = retrieve_to(dataset=halueval, key="hallucination")

    for i in range(len(queries)):

        print(f"{i + 1}/{len(queries)}\n")
        print(f'Query: {queries[i]}\n-------ChatGPT Response-------\n{response[i]}\nHallucination: {tag[i]}')

        results = {}

        for x in range(2):
            
            result = return_results( # Change these settings if you want to test them on your own
                model="meta-llama/Llama-3.2-3B-Instruct-Turbo",
                prompt=queries[i],
                temperature=1.1,
                logprobs=5,
                top_k=10,
                top_p=0.9,
                max_tokens=150,
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
            result["label"] = label
            
            results[x] = result



        with open("AnsDataset.jsonl", "a", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)
            f.write("\n")
        print("---Successfully stored---\n")