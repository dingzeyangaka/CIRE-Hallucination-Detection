# CIRE

Core implementation of **Grounded or Fabricated? Unsupervised Detection of LLM Hallucinations via Contextualized Influence on Response Embeddings**.

How to use it (Python 3.10+):

```bash
pip install -r requirements.txt
python main.py
```

Use another model or layer:

```bash
python main.py --model /path/to/model --layer 16
```

```python
from CIRE import CIRE

scorer = CIRE(model, tokenizer, layer=16)
scores = scorer.score("Question: What is the capital of France?\nAnswer:", " Paris.")
print(scores["cire"], scores["risk"])
```

`CIRE = ||H_R - H_QR||_F / ||H_R||_F * exp(mean(token_log_probs))`; risk is `-CIRE`, not a probability. Both passes use the same response tokens; their conditional probabilities are computed by teacher forcing.

`example.json` contains eight hand-written pairs (`label=1`: hallucination). The small default model demonstrates usage; it does not reproduce the paper's benchmark results.
