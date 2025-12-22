from together import Together
from typing import List
import json

client = Together(api_key="tgp_v1_A7ddbftsUdpKjhmtzMQBplkDPXP3uI82cYTgbh5migo")

def get_embeddings(texts: List[str], model: str) -> List[List[float]]:
  texts = [text.replace("\n", " ") for text in texts]
  outputs = client.embeddings.create(model=model, input = texts)
  return [outputs.data[i].embedding for i in range(len(texts))]

input_texts = ['Our solar system orbits the Milky Way galaxy at about 515,000 mph']
embeddings = get_embeddings(input_texts, model='togethercomputer/m2-bert-80M-8k-retrieval')

def return_results(
        model: str,
        prompt: str,
        logprobs: int | None = 5,
        temperature: float | None = 1.0,
        max_tokens: int | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        ):
  
  response = client.chat.completions.create(
    model=model,
    messages=[
      {
        "role": "system",
        "content": "You are a helpful assistant that will answer the question as simple as possible. You will answer within 50 tokens."
      },
      {
        "role": "user", 
        "content": prompt
      },
    ],
    max_tokens=max_tokens,
    logprobs=logprobs,
    temperature=temperature,
    top_k = top_k,
    top_p = top_p,
  )

  response_logprobs = response.choices[0].logprobs.top_logprobs
  response_content = response.choices[0].message.content

  res = {
    "generated_text": response_content,
    "top_logprobs": response_logprobs,
  }

  return res

if __name__ == "__main__":
  with open("generation_results.json", "w", encoding="utf-8") as out_f:
          json.dump(return_results("meta-llama/Llama-3.2-3B-Instruct-Turbo", "Generate some random English words"), out_f, indent=2, ensure_ascii=False)


