# Matched reader protocol

Take the first 100 official development IDs per dataset ordered by SHA-256 of UTF-8 `graphmind-v1-answer:` plus question ID, breaking ties by original ID. Compare dense, hybrid and graph-2 retrieval with one frozen FLAN-T5-small reader. Its revision, seven verified upstream file hashes and redistributable tokenizer are in `benchmarks/vendor/flan/`. Each question retains the original supplied candidate set. A split's canonical cases, judgments, full membership and source audit are separately bound to digests derived from the verified official source files in `benchmarks/vendor/canonical.json`.

Each method supplies its first four distinct paragraphs in rank order. The prompt starts with “Answer the question briefly. Question: ” and the question (the entire line has at most 118 tokens), followed by a newline, “Context:” and a newline. Each paragraph starts with “Title: ” and its original title, then a newline, “Passage: ” and its original body. The prefix of each paragraph block is independently capped at 96 tokens including its labels and title. Unused budget stays unused. Token offsets give a candidate prefix, shortened by Unicode code point until its re-encoding fits. Newlines separate blocks. The final line is “Answer:”. Both the entire context (384 tokens maximum) and final text (512 tokens maximum including EOS) are retokenized and checked. Each recorded source slice starts at zero and ends at its declared title/body end. Body positions in the final prompt are half-open Unicode code-point offsets. The tokenizer is the pinned fast tokenizer JSON with its upstream normalization and EOS postprocessor.

Before any reader weights are loaded, the runner verifies the complete retrieval manifest, binds the local measurement code to Git, verifies the FLAN files, and writes the 600 method-specific prompts, input IDs, source spans, selection and binding. The CPU float32 reader uses eager attention, six intra-op threads, one inter-op thread, deterministic algorithms, evaluation mode, one example per call, greedy decoding, one beam and a 48-new-token cap. A verified-only temporary directory supplies local safetensors weights. The record includes decoder start ID 0 followed by the generated IDs ending at the first EOS (1) or the length limit. Decode with special tokens skipped using the published tokenizer. Generated output is scored separately after inference.

The primary outcome is graph-2 minus hybrid reference-token F1. In each dataset use paired question percentile bootstrap (10,000 draws, PCG64 seed 20261119), two-sided paired sign-flip randomization (10,000 draws, fresh same seed, plus-one correction; numerical tie tolerance 1e-12 on the summed delta) and Holm adjustment across the two datasets. Supplementary cluster intervals resample occupied full-development support-connected components uniformly with replacement and weight by the sampled number of selected questions. Fewer than two occupied components gives an unavailable interval. EM and contrasts involving dense are descriptive. Per-question scores are independently compared with the two author scorers, retaining aliases and their dataset-specific empty/yes/no handling.

The evaluation measures normalized string overlap with references. It does not adjudicate semantic faithfulness. Pretraining exposure to public questions or Wikipedia is possible. Short contexts, a small reader and 100 selected questions per dataset limit inference. Shared articles and subquestions can introduce dependence beyond exact shared support paragraphs. Recorded timing and model outputs are observations. Model-free replay checks artifact consistency and cannot authenticate historical execution.

## Recorded loader behavior

Transformers 5.23.0 emitted the following warning during the original load.

> The tied weights mapping and config for this model specifies to tie shared.weight to lm_head.weight, but both are present in the checkpoints with different values, so we will NOT tie them. You should update the config with `tie_word_embeddings=False` to silence this warning.

The experiment retained the upstream configuration. After the run, a separate fresh load of the same verified files produced an empty missing/unexpected/mismatched/error report. All 192 tensors in the loaded state matched the dtype and values in the 190 checkpoint entries, including two embedding aliases. The shared and LM-head tensors matched their distinct checkpoint values. The exact hashes, warning and timing of this independent reload are in `results/answers/weight-audit.json`. It generated zero predictions. Future runs perform the same tensor check before generation.

## Replay and reproduction

Run these commands from a Git checkout with the original commit history available. The verifier checks source digests against both frozen Git trees. CI checkouts need full history.

```sh
python -m pip install -e . -r requirements-replay.txt
python -m benchmarks.verify
```

The verifier rebuilds canonical case/judgment bindings, membership, sparse scores, graph expansions and training selections, and checks each dense score against independently multiplied stored vector coordinates. On every split it recalculates all ranks, per-question metrics, aggregates, independent trec_eval checks, primary paired estimates and timing summaries. It reconstructs the 200 ordered four-passage inputs for each answer method from complete original paragraphs, verifies input token IDs and source spans, decodes every output through its EOS or length stop, reruns both dataset scorers and reconstructs the confidence intervals. It also regenerates the 200 explorer graphs with Unicode mention spans. Original measurements and the explicitly post-run retrieval-source and weight-loader audits remain independently identified.

```sh
python -m pip install -e . -r requirements-benchmark.txt
scripts/reproduce.sh reproductions/run-01 data/cache
python -m benchmarks.verify --root reproductions/run-01 --reproduction
```

The reproduction downloads and checks the pinned datasets and both model checkpoints, reruns the experiments, and publishes a separately validated directory. Existing output paths are rejected. Private reproduction observations remain beside the release for diagnosis. CPU hardware, framework versions and floating point can affect new observations.
