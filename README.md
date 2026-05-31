# **AP Research Project Code**

#### **_LLM Hallucination detection based on Cosine Similarity and Token Entropy_**

These are the code and database used for the AP Research project. The aim of the project is to detect hallucinations of LLMs (Large Language Models) by using logit outputs and the generated text only, a standard white box approach to hallucination detection. 

## Methodology

In this study we choose to use the questions in TruthfulQA as the queries. The model this study chooses is Llama-3.2-3B-Instruct-Turbo. We repeat every query two times and collect the answers from Llama-3.2, tagging all the hallucinated answers. 

Different from previous methods, this study tries to use neural networks to evaluate the output of the LLM. Two models are constructed, one is a MLP model that analyzes only the perplexity and the cosine similarity of the two generated text. The other model consists of an attention layer aimed to analyze the difference in the two answer's logit for every token. The model takes in two logit matrices, perplexity, and cosine similarity. Both models will output a float from 0 to 1 indicating the likelyhood of the texts being a hallucination. An output of 0.5 and above is identified as a hallucination, while anything below is identified as a hallucination. If any two answers from the llm contain a hallucination, the model output of 0.5 and above is seen as the correct answer, vice versa.

## Justification

The underlying logic behind all this is that when models hallucinate, they tend to answer the same question differently when asked multiple times. The cosine similarity analyzes how the two models differ in semantics while the attention layer in the first model analyzes the difference of the logits. Perplexity indicates how much the model trusts its output, showing for each token, the output token is chosen from how many tokens. In other words, how confused the LLM is when generating the answer token. The two models exploited these data to output a halllucination percentage.

### Credits

Farquhar, Sebastian, et al. “Detecting Hallucinations in Large Language Models Using Semantic Entropy.” Nature, vol. 630, no. 8017, June 2024, pp. 625–30.       Crossref, https://doi.org/10.1038/s41586-024-07421-0.

Li, Junyi, et al. “HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models.” arXiv:2305.11747, arXiv, 23 Oct. 2023. arXiv.org, https://doi.org/10.48550/arXiv.2305.11747.

Manakul, Potsawee, et al. “SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models.” arXiv:2303.08896, arXiv, 11 Oct. 2023. arXiv.org, https://doi.org/10.48550/arXiv.2303.08896.

RUCAIBox. “GitHub - RUCAIBox/HaluEval: This Is the Repository of HaluEval, a Large-Scale Hallucination Evaluation Benchmark for Large Language Models.” GitHub, 2025, github.com/RUCAIBox/HaluEval.

Seth. Deep Learning from Scratch : Building with Python from First Principles. O’reilly Media, Inc, 2019.

Sriramanan, Gaurang, et al. LLM-Check: Investigating Detection of Hallucinations in Large Language Models.
Vaswani, Ashish, et al. “Attention Is All You Need.” arXiv:1706.03762, arXiv, 2 Aug. 2023. arXiv.org, https://doi.org/10.48550/arXiv.1706.03762.

StatQuest with Josh Starmer. “Coding a ChatGPT like Transformer from Scratch in PyTorch.” YouTube, 30 June 2024, www.youtube.com/watch?v=C9QSpl5nmrY. Accessed 11 Jan. 2026.Weidman, 
