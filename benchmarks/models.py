"""Offline float32 CPU MiniLM and replayable embedding tables."""

import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .sources import MODEL, verified

REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
THREADS = 6
BATCH = 32
LIMIT = 256
WIDTH = 384


def model_identity():
    settings = {
        "revision": REVISION,
        "files": {Path(s["path"]).name: s["sha256"] for s in MODEL},
        "max_tokens": LIMIT,
        "pool": "attention-mask mean then L2",
        "dtype": "float32",
        "device": "cpu",
        "threads": THREADS,
        "interop_threads": 1,
        "batch": BATCH,
        "attention": "sdpa",
        "input": "title LF body / raw question",
    }
    return settings, hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


@contextmanager
def load_encoder(cache):
    # Private, freshly created directory contains only regular verified copies; a
    # stray tokenizer or custom module in the download cache cannot enter a loader.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(THREADS)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(THREADS)
    os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS)
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with tempfile.TemporaryDirectory() as root:
        for spec in MODEL:
            name = Path(spec["path"]).name
            dest = Path(root) / name
            shutil.copyfile(cache / "minilm" / name, dest)
            verified(dest, spec)
        tokenizer = AutoTokenizer.from_pretrained(
            root, local_files_only=True, trust_remote_code=False
        )
        network = (
            AutoModel.from_pretrained(
                root,
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                dtype=torch.float32,
                attn_implementation="sdpa",
            )
            .to("cpu")
            .eval()
        )
        yield Encoder(tokenizer, network)


class Encoder:
    def __init__(self, tokenizer, network):
        self.tokenizer = tokenizer
        self.network = network

    def encode(self, texts):
        import numpy as np
        import torch

        # Tokenizing without a limit gives the original lengths. Truncate via
        # tokenizer so the final special token is preserved at the cap.
        lengths = [
            len(x) for x in self.tokenizer(texts, truncation=False, verbose=False)["input_ids"]
        ]
        order = sorted(range(len(texts)), key=lambda i: (min(lengths[i], LIMIT), i))
        vectors = np.empty((len(texts), WIDTH), dtype="<f4")
        with torch.inference_mode():
            for start in range(0, len(order), BATCH):
                idx = order[start : start + BATCH]
                batch = self.tokenizer(
                    [texts[i] for i in idx],
                    padding=True,
                    truncation=True,
                    max_length=LIMIT,
                    return_tensors="pt",
                    verbose=False,
                )
                hidden = self.network(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                vectors[idx] = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()
        if not np.isfinite(vectors).all():
            raise ValueError("nonfinite embeddings")
        return vectors, lengths


def cosine_scores(ids, docs, query):
    import numpy as np

    # Fixed reduction across each contiguous 384-coordinate row, independent of
    # vendor GEMM choices. Includes zero and negative cosine scores.
    scores = np.sum(docs * query, axis=1, dtype=np.float32)
    return dict(zip(ids, map(float, scores), strict=True))
