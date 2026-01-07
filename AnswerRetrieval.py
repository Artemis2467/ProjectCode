from together import Together
import os
import json

client = Together(api_key=os.environ["TOGETHER_API_KEY"])

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
        "content": "answer the questions as short as possible. Avoid using any form of code."
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
  if logprobs:
    response_logprobs = response.choices[0].logprobs.top_logprobs
  response_content = response.choices[0].message.content

  res = {
    "generated_text": response_content,
    "top_logprobs": response_logprobs,
  }

  return res

if __name__ == "__main__":
  with open("generation_results.json", "w", encoding="utf-8") as out_f:
          json.dump(
              return_results("meta-llama/Llama-3.2-3B-Instruct-Turbo", "Generate some random English words"),
              out_f,
              indent=2,
              ensure_ascii=False
              )