"""
Second check for reviews written by modern AI (ChatGPT, GPT-4, LLaMA and similar).

The project's own models were trained on GPT-2-era fake reviews and do not
recognise fluent modern AI text. This module runs a pretrained transformer
detector for that: the Longformer released with the MAGE benchmark
("MAGE: Machine-generated Text Detection in the Wild", Li et al. 2024),
https://huggingface.co/nealcly/detection-longformer, trained on human text and
text from 27 language models.

The model (about 600 MB) is downloaded from Hugging Face on first use and cached.
It needs the optional packages torch and transformers; without them the app
simply skips this check. It is slow on a CPU, about 1 second per review.
"""
import os

REPO = "nealcly/detection-longformer"
AI_INDEX = 0          # output 0 = machine-generated, 1 = human-written (MAGE's convention)
MAX_TOKENS = 512


class ModernAIDetector:
    def __init__(self):
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging
        logging.set_verbosity_error()
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(REPO)
        self.model = AutoModelForSequenceClassification.from_pretrained(REPO).eval()

    def predict(self, texts, batch_size=8, progress=None):
        """Probability that each text was written by a modern AI model."""
        texts = [t if isinstance(t, str) else "" for t in texts]
        out = []
        for start in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[start:start + batch_size], truncation=True, max_length=MAX_TOKENS,
                                 padding=True, return_tensors="pt")
            with self.torch.no_grad():
                out.extend(self.torch.softmax(self.model(**enc).logits, dim=-1)[:, AI_INDEX].tolist())
            if progress is not None:
                progress(min(start + batch_size, len(texts)), len(texts))
        return out


def load_modern_detector():
    """The detector, or None if torch/transformers are missing or the model can't be downloaded."""
    try:
        return ModernAIDetector()
    except Exception:
        return None
