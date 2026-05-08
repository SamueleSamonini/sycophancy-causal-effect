"""
Model inference utilities for the Sycophancy Causal Effect study.

Provides logit-based scoring of binary (A/B) multiple-choice answers,
which works identically on base and instruction-tuned models without
chat templates, isolating the causal effect of instruction tuning from
prompt-format confounders.
"""

from __future__ import annotations
from typing import Optional, Dict
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def score_agreement(
    model,
    tokenizer,
    prompt: str,
    agree_token: str = "A",
    disagree_token: str = "B",
) -> Dict[str, float]:
    """
    Compute P(next_token = agree_token) vs P(next_token = disagree_token)
    via softmax over the two-choice subspace.

    Works identically on base and instruction-tuned models because it does
    not rely on chat templates, isolating the causal effect of instruction
    tuning from prompt format.

    Args:
        model: HuggingFace causal LM (already loaded on a device).
        tokenizer: corresponding tokenizer.
        prompt: input text ending with a multiple-choice request (A/B).
        agree_token: token signaling agreement (default "A").
        disagree_token: token signaling disagreement (default "B").

    Returns:
        Dict with normalized probabilities `p_agree` and `p_disagree` (sum to 1).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Logits at the last position predict the next token
    next_token_logits = outputs.logits[0, -1, :]

    # Token IDs for "A" and "B" (with leading space, typical for BPE tokenizers)
    agree_id = tokenizer.encode(" " + agree_token, add_special_tokens=False)[0]
    disagree_id = tokenizer.encode(" " + disagree_token, add_special_tokens=False)[0]

    # Softmax restricted to the two relevant tokens
    relevant_logits = next_token_logits[[agree_id, disagree_id]]
    probs = torch.softmax(relevant_logits, dim=0)

    return {
        "p_agree": probs[0].item(),
        "p_disagree": probs[1].item(),
    }


class ModelScorer:
    """
    Encapsulates a HuggingFace causal LM together with its tokenizer for
    sycophancy scoring. Manages loading, scoring, and VRAM cleanup.

    Example:
        scorer = ModelScorer.load("Qwen/Qwen2.5-1.5B", cache_dir=CACHE_DIR)
        result = scorer.score_agreement(prompt)
        scorer.unload()  # free VRAM when done
    """

    def __init__(self, name: str, model, tokenizer):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def load(
        cls,
        model_name: str,
        cache_dir: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ) -> "ModelScorer":
        """
        Load a model and its tokenizer in fp16 with automatic device mapping.
        
        Uses low_cpu_mem_usage=True to load weights directly onto the GPU without
        an intermediate full copy in CPU RAM, which is critical on Colab Free
        (~12 GB RAM) when loading multiple models in sequence.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
        )
        return cls(name=model_name, model=model, tokenizer=tokenizer)

    def score_agreement(
        self,
        prompt: str,
        agree_token: str = "A",
        disagree_token: str = "B",
    ) -> Dict[str, float]:
        """Score a prompt using this scorer's model and tokenizer."""
        return score_agreement(
            self.model,
            self.tokenizer,
            prompt,
            agree_token=agree_token,
            disagree_token=disagree_token,
        )

    def unload(self) -> None:
        """Release the model from VRAM (useful when switching between large models)."""
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __repr__(self) -> str:
        return f"ModelScorer(name={self.name!r})"