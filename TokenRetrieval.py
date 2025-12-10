import json
import math
import requests
from typing import List, Dict, Optional

API_URL = "https://api.together.xyz" 
API_KEY = "tgp_v1_A7ddbftsUdpKjhmtzMQBplkDPXP3uI82cYTgbh5migo"
MODEL = "meta-llama/Llama-3.2-3B-Instruct-Turbo" 
if not API_URL or not API_KEY or not MODEL:
    raise SystemExit(
        "Set API_URL, API_KEY and MODEL environment variables before running.\n"
        "Example:\n"
        "  export API_URL=https://api.together.ai\n"
        "  export API_KEY=sk-...\n"
        "  export MODEL=llama-2-7b\n"
    )

PROMPT = (
    "Who are you?"
)

MAX_TOKENS = 150
TEMPERATURE = 0.2
TOP_LOGPROBS = 5 

def stable_softmax(logits: List[float]) -> List[float]:
    """Compute softmax from logits in a numerically stable way."""
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    if s == 0:
        # avoid division by zero; return uniform small probabilities
        n = len(logits)
        return [1.0 / n] * n
    return [e / s for e in exps]

def safe_exp(logp: Optional[float]) -> Optional[float]:
    """Exponentiate a log-probability safely; returns None if input is None."""
    if logp is None:
        return None
    # clamp to avoid underflow to 0 for very small logprobs
    try:
        p = math.exp(logp)
        return p
    except OverflowError:
        return 0.0

def return_result(max_tokens: int, temperature: float, logprobs: int, prompt: str, model: str = MODEL) -> dict:
    """
    Send a completion request and return the parsed results dict.

    Parameter order: max_tokens, temperature, logprobs, prompt
    """
    endpoint = f"{API_URL.rstrip('/')}/v1/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "logprobs": logprobs,
    }

    resp = requests.post(endpoint, headers=headers, json=payload, timeout=120)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print("Request failed:", e)
        print("Status:", resp.status_code)
        try:
            print("Response JSON:", resp.json())
        except Exception:
            print("Response body:", resp.text)
        raise

    data = resp.json()
    if "choices" not in data or len(data["choices"]) == 0:
        raise SystemExit("No choices returned by the API. Full response:\n" + json.dumps(data, indent=2))

    choice = data["choices"][0]

    # Extract generated text from common locations
    generated_text: str = ""
    if isinstance(choice, dict):
        # OpenAI-style completion
        generated_text = choice.get("text", "")
        # Chat-style content
        if not generated_text and isinstance(choice.get("message"), dict):
            generated_text = choice["message"].get("content", "")
        # Other possible field
        if not generated_text and "output_text" in choice:
            generated_text = choice.get("output_text", "")
    if not generated_text:
        # last resort: try top-level fields
        generated_text = choice.get("text", "") or choice.get("content", "") or ""

    # Variables to populate
    tokens: List[str] = []
    token_logprobs: List[Optional[float]] = []
    token_probs: List[Optional[float]] = []
    top_logprobs: List[Dict[str, float]] = []

    # Case A: OpenAI-style 'logprobs' present
    if "logprobs" in choice and choice["logprobs"] is not None:
        lp = choice["logprobs"]
        tokens = lp.get("tokens", [])
        token_logprobs = lp.get("token_logprobs", [])
        top_logprobs = lp.get("top_logprobs", [])
        token_probs = [safe_exp(l) for l in token_logprobs]

    # Case B: Provider returned logits per step
    elif "logits" in choice and choice["logits"] is not None:
        logits = choice["logits"]
        tokens = choice.get("tokens", [])
        token_probs = []
        token_logprobs = []
        for i, step_logits in enumerate(logits):
            probs = stable_softmax(step_logits)
            chosen_id = None
            if "token_ids" in choice:
                chosen_id = choice["token_ids"][i]
            elif i < len(tokens) and isinstance(tokens[i], str):
                chosen_id = None
            if chosen_id is None:
                token_probs.append(None)
                token_logprobs.append(None)
                top_logprobs.append({})
            else:
                p = probs[chosen_id]
                token_probs.append(p)
                token_logprobs.append(math.log(p) if p > 0 else float("-inf"))
                topk = sorted(enumerate(step_logits), key=lambda x: x[1], reverse=True)[:logprobs]
                top_logprobs.append({str(idx): float(logit) for idx, logit in topk})
    else:
        # No token-level info returned; save raw response for debugging
        print("Provider did not return 'logprobs' or 'logits'. Full response saved to 'raw_response.json'.")
        with open("raw_response.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        generated_text = generated_text or choice.get("text", "")
        tokens = []
        token_logprobs = []
        token_probs = []
        top_logprobs = []


    # for i, tok in enumerate(tokens):
    #     lp = token_logprobs[i] if i < len(token_logprobs) else None
    #     p = token_probs[i] if i < len(token_probs) else None
    #     topk = top_logprobs[i] if i < len(top_logprobs) else None
    #     print(f"{i:3d}: token={repr(tok):30s}  logprob={lp!s:12s}  prob={p!s:12s}  topk={topk}")

    results = {
        "generated_text": generated_text, # 5000 * 100 * 3
        "tokens": tokens, # 5000 * 100 * 3
        "token_probs": token_probs, # 5000 * 100 * 3
        "top_logprobs": top_logprobs, # 5000 * 100 * 5 * 3
    }

    return results


def return_results(prompt: str = PROMPT, max_tokens: int = MAX_TOKENS, temperature: float = TEMPERATURE, logprobs: int = TOP_LOGPROBS, model: str = MODEL) -> dict:
    return return_result(max_tokens=max_tokens, temperature=temperature, logprobs=logprobs, prompt=prompt, model=model)


if __name__ == "__main__":
    res = return_result(MAX_TOKENS, TEMPERATURE, TOP_LOGPROBS, PROMPT)
    with open("generation_results.json", "w", encoding="utf-8") as out_f:
        json.dump(res, out_f, indent=2, ensure_ascii=False)