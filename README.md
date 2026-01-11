# **AP Research Project Code**

#### **_LLM Hallucination detection based on Cosine Similarity and Token Entropy_**

These are the code and database used for the AP Research project. The aim of the project is to detect hallucinations of LLMs (Large Language Models) by using logit outputs and the generated text only, a standard grey box approach to hallucination detection. 

## Methodology

In this study we choose to use the HaluEval's general_data dataset as the queries. The model this study chooses is Llama-3.2-3B-Instruct-Turbo. We repeat every query two times and collect the answers from Llama-3.2, tagging all the hallucinated answers. 

Different from previous methods, this study tries to use neural networks to evaluate the output of the LLM. Two models are constructed, one is a Linear model that analyzes only the perplexity and the cosine similarity of the two generated text. The other mode consists of an attention layer aimed to analyze the difference in the two answer's logit for every token. The model takes in two logit matrices, perplexity and cosine similarity. Both models will output a float from 0 to 1 indicating the likely hood of the texts being a hallucination. An output of 0.5 and above is identified as a hallucination, while anything below is identified as a hallucination. If any two answers from the llm contain a hallucination, the model output of 0.5 and above is seen as a correct answer, vice versa.

## Justification

The underlying logic behind all this is that when models hallucinate, they tend to answer the same question differently when asked multiple times. The cosine similarity analyzes the how the two models differ in semantics while the attention layer in the first model analyzes the difference of the logits. Perplexity indicates how much the model trusts its output, showing for each token, the output token is chosen from how many tokens. In other words, how confused the LLM is when generating the answer token.
