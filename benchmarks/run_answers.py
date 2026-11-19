"""Prepare immutable matched reader inputs, then run greedy CPU generation."""

import argparse
import os
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path

from .generation import FLAN, SETTINGS, analyze, check_output, tokenizer
from .release import (
    answer_inputs,
    bound_rows,
    check_inventory,
    judgments,
    read,
    verify_source_files,
)
from .run_retrieval import compressed, json_bytes, platform_info, put_json, write_shards
from .sources import destination, digest, source_binding, verified


def fetch_flan(cache):
    for name, pin in FLAN["files"].items():
        path = cache / "flan" / name
        if not path.exists():
            url = f"https://huggingface.co/{FLAN['model']}/resolve/{FLAN['revision']}/{name}"
            with (
                destination(path, pin) as temp,
                urllib.request.urlopen(url, timeout=120) as response,
                temp.open("wb") as f,
            ):
                shutil.copyfileobj(response, f)
        verified(path, pin)


def run(cache, retrieval, root):
    checkout = Path(__file__).resolve().parents[1]
    binding = source_binding(checkout)
    verify_source_files()
    check_inventory(retrieval, read(retrieval / "manifest.json")["files"], ("manifest.json",))
    manifest_sha = digest(retrieval / "manifest.json")
    for name, spec in FLAN["files"].items():
        verified(cache / "flan" / name, spec)
    tok = tokenizer()
    ids, inputs = answer_inputs(retrieval, tok)
    for ds, rows in inputs.items():
        write_shards(root / ds / "inputs", rows)
    put_json(root / "selection.json", ids)
    os.environ.update(HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="6")
    import torch
    from transformers import AutoModelForSeq2SeqLM, GenerationConfig

    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    opts = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=48,
        decoder_start_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
        use_cache=True,
    )
    put_json(
        root / "binding.json",
        {
            "schema": 1,
            "assembly": "pre-model-load",
            "source_binding": binding,
            "retrieval_manifest_sha256": manifest_sha,
            "model": FLAN,
            "settings": SETTINGS,
            "tokenizer": "tokenizers.Tokenizer.from_file (pinned upstream tokenizer.json)",
            "generation_config": opts.to_dict(),
            "inputs_sha256": {ds: bound_rows(rows) for ds, rows in inputs.items()},
            "hardware": platform_info(),
        },
    )
    print("Bound 600 inputs and measurement sources before loading model.", flush=True)
    started = time.monotonic()
    predictions = {}
    with tempfile.TemporaryDirectory() as verified_dir:
        for name, pin in FLAN["files"].items():
            dest = Path(verified_dir) / name
            shutil.copyfile(cache / "flan" / name, dest)
            verified(dest, pin)
        net = (
            AutoModelForSeq2SeqLM.from_pretrained(
                verified_dir,
                local_files_only=True,
                use_safetensors=True,
                trust_remote_code=False,
                dtype=torch.float32,
                attn_implementation="eager",
            )
            .to("cpu")
            .eval()
        )
        with torch.inference_mode():
            for ds, rows in inputs.items():
                predictions[ds] = []
                for i, r in enumerate(rows):
                    tokens = torch.tensor([r["input_ids"]], dtype=torch.long)
                    generated = net.generate(
                        input_ids=tokens,
                        attention_mask=torch.ones_like(tokens),
                        generation_config=opts,
                    )
                    output = generated[0].tolist()
                    predictions[ds].append(
                        {
                            "qid": r["qid"],
                            "method": r["method"],
                            "decoder_ids": output,
                            **check_output(tok, output),
                        }
                    )
                    if (i + 1) % 64 == 0 or i + 1 == len(rows):
                        chunk = predictions[ds][(i // 64) * 64 : i + 1]
                        compressed(
                            root / ds / "predictions" / f"{i // 64:04}.jsonl.gz",
                            b"".join(json_bytes(p) + b"\n" for p in chunk),
                        )
                        print(f"{ds}: retained {i + 1}/{len(rows)} predictions", flush=True)
    pairs, summaries = analyze(predictions, judgments(retrieval), ids)
    for ds, rows in pairs.items():
        write_shards(root / ds / "metrics", rows)
    put_json(root / "summary.json", summaries)
    put_json(
        root / "elapsed.json",
        {"model_load_inference_and_scoring_seconds": time.monotonic() - started},
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=Path("data/cache"))
    p.add_argument("--retrieval", type=Path, default=Path("results/retrieval"))
    p.add_argument("--output", type=Path)
    p.add_argument("--download-only", action="store_true")
    args = p.parse_args()
    if args.download_only:
        fetch_flan(args.cache)
        return
    if not args.output or args.output.exists():
        p.error("provide a new output directory")
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # Incomplete work is private, outside the release tree, on the same filesystem.
    private = out.parent.parent / ".graphmind-staging"
    private.mkdir(exist_ok=True)
    lock = private / (out.name + ".lock")
    with lock.open("x"):
        try:
            stage = Path(tempfile.mkdtemp(dir=private, prefix=out.name + "-"))
            print(f"Staging in {stage.name}", flush=True)
            run(args.cache, args.retrieval, stage)
            if out.exists():
                raise FileExistsError(out)
            stage.rename(out)
        finally:
            lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
