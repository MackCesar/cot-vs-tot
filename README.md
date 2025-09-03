# CoT vs ToT (Reasoning with LLMs)

<p>This project explores different reasoning strategies for large language models (LLMs):</p>

 • **Chain-of-Thought (CoT)** → a single reasoning path, step by step.\
 • **Self-Consistency** → sample multiple CoTs, then vote for the most consistent answer.\
 • **Tree-of-Thoughts (ToT)** → branching exploration of possible reasoning steps. \
 • **Graph-of-Thoughts (GoT)** → voting across multiple CoT runs (a toy graph-based variant).

<p>The idea is to see how prompting strategies can improve reasoning accuracy on benchmarks like GSM8K, BIG-Bench Hard, and ARC-Challenge. This repo is a research playground for myself: it’s not about state-of-the-art results, but about learning how reasoning methods differ in practice.</p>

## What you’ll learn

• How prompting alone can change the reasoning ability of LLMs. \
• Why sampling multiple paths (self-consistency) often beats a single chain-of-thought. \
• How to set up a reproducible evaluation harness for reasoning tasks. \
• How to run simple experiments with OpenAI or local HuggingFace models. 

### Requirements

This project requires Python 3.10 or higher. To install dependencies, run:
 ```bash
 pip install -r requirements.txt
 ```
 
You will also need API keys: \
• OpenAI API key: set it in a .env file as OPENAI_API_KEY=… \
• (Optional) Hugging Face token: set HUGGINGFACEHUB_API_TOKEN=… if you want to use local transformers models.

### How to run

To run a simple Chain‑of‑Thought experiment on the GSM8K dataset:
```python
python -m src.cot --dataset openai/gsm8k --subset main --split test --max-samples 20 --model openai:gpt-4o-mini
```
To try the Graph‑of‑Thoughts variant:
```python
python -m src.graph_of_thought --dataset openai/gsm8k --subset main --split test --max-samples 10 --iters 4 --model openai:gpt-4o-mini
```
Results are saved under the results/ directory, and you can compute accuracy with:
```bash
python -m src.eval --run-dir results/<your_run_folder>
```
### Key takeaways
• Reasoning is not “one prompt and done.” Different strategies yield different results. \
• Self‑consistency and tree‑based approaches can uncover better reasoning paths than plain chain‑of‑thought. \
• Prompting alone can provide research‑level insights without any model fine‑tuning.

### References
• [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)\
• [Self-Consistency](https://arxiv.org/abs/2203.11171)
• [Tree-of-Thoughts](https://arxiv.org/abs/2305.10601)
• [BIG-Bench Hard](https://arxiv.org/abs/2206.04615)

### License
This project is licensed under the [MIT License](./LICENSE).