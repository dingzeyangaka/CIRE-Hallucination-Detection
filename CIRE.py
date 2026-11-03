"""CIRE: contextualized influence on response embeddings."""

import torch


def cire_score(answer_only, conditioned_answer, token_log_probs):
    """Compute IFL, mean token probability, CIRE, and risk for aligned tokens."""
    a = torch.as_tensor(answer_only).float()
    b = torch.as_tensor(conditioned_answer, device=a.device).float()
    log_p = torch.as_tensor(token_log_probs, device=a.device).float()
    if a.ndim != 2 or a.shape != b.shape or a.numel() == 0:
        raise ValueError("Answer representations must share a nonempty (tokens, hidden) shape.")
    if log_p.ndim != 1 or log_p.numel() != a.shape[0]:
        raise ValueError("Provide one conditional log probability per response token.")
    if not all(torch.isfinite(x).all() for x in (a, b, log_p)) or (log_p > 0).any():
        raise ValueError("Inputs must be finite; log probabilities must be nonpositive.")
    denominator = torch.linalg.vector_norm(a)
    if denominator == 0:
        raise ValueError("The answer-only representation must have a nonzero norm.")
    ifl = (torch.linalg.vector_norm(a - b) / denominator).item()
    probability = log_p.mean().exp().item()
    return {"ifl": ifl, "probability": probability,
            "cire": ifl * probability, "risk": -ifl * probability}


class CIRE:
    def __init__(self, model, tokenizer, layer=None):
        """Use a Hugging Face causal LM; layer indices start at 1."""
        self.model = model.eval()
        self.tokenizer = tokenizer
        n_layers = model.config.num_hidden_layers
        self.layer = max(1, n_layers // 2) if layer is None else layer
        if not 1 <= self.layer <= n_layers:
            raise ValueError(f"layer must be between 1 and {n_layers}.")

    @torch.inference_mode()
    def score(self, prompt, response):
        """Score a fixed response under a fully formatted question prompt."""
        if not prompt.strip() or not response.strip():
            raise ValueError("prompt and response must be nonempty strings.")
        device = self.model.get_input_embeddings().weight.device
        prefix = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        answer = self.tokenizer(response, add_special_tokens=False,
                                return_tensors="pt").input_ids.to(device)
        if prefix.numel() == 0 or answer.numel() == 0:
            raise ValueError("prompt and response must contain at least one token.")
        # Reuse exactly the same response token IDs in both forward passes.
        answer_only = answer
        bos = self.tokenizer.bos_token_id
        if bos is not None:
            answer_only = torch.cat((answer.new_tensor([[bos]]), answer), dim=1)
        conditioned = torch.cat((prefix, answer), dim=1)
        limit = getattr(self.model.config, "max_position_embeddings", None)
        if limit and max(answer_only.shape[1], conditioned.shape[1]) > limit:
            raise ValueError(f"Input exceeds the model's {limit}-token context window.")
        n_tokens = answer.shape[1]
        output = self.model(input_ids=answer_only,
                            attention_mask=torch.ones_like(answer_only),
                            output_hidden_states=True, use_cache=False, return_dict=True)
        a = output.hidden_states[self.layer][0, -n_tokens:].float().cpu()
        del output
        output = self.model(input_ids=conditioned,
                            attention_mask=torch.ones_like(conditioned),
                            output_hidden_states=True, use_cache=False, return_dict=True)
        b = output.hidden_states[self.layer][0, -n_tokens:].float().cpu()
        # The position immediately before each response token predicts that token.
        start = prefix.shape[1] - 1
        logits = output.logits[0, start:start + n_tokens].float()
        log_p = logits.log_softmax(dim=-1).gather(1, answer[0, :, None]).squeeze(1).cpu()
        return cire_score(a, b, log_p)
