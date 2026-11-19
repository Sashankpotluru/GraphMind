# Supplied-candidate retrieval protocol

The unit of evaluation is one question with its supplied paragraphs. Both datasets include relevant evidence in the candidates by construction. Each system ranks every candidate. Results describe the official development splits. They do not measure open-Wikipedia search, answer accuracy, or hidden-test performance.

## Inputs and split membership

The digest and size manifest in `benchmarks/sources.json` identifies HotpotQA distractor from the official Hugging Face revision `1908d6afbbead072334abe2965f91bd2709910ab` and MuSiQue-Ans v1.0 from the authors' archive. Archive extraction streams only the two named, size-bounded members into fixed destinations. Before measuring, the runner checks each source SHA-256, including the archive and extracted files. The input row counts must be 90,447/7,405 and 19,938/2,417, respectively, for training/development.

The training sample comprises the first 500 IDs per dataset ordered by SHA-256 of UTF-8 `graphmind-v1-tune:` plus the original ID, breaking ties by ID. Every development ID enters the evaluation in ascending ID order. Duplicate IDs within a split or across splits invalidate the run. Per-dataset inventories list every ID. Audits report repeated NFKC/casefold/whitespace-normalized questions and intersecting exact paragraph IDs for full training versus development and selected training versus development. Shared articles and recurring subquestions can still introduce dependence.

Hotpot bodies concatenate the original sentence strings without inserting characters. An annotated title selects every source paragraph with that exact title. Some supplied sentence indices fall outside their title's sentence array. Their existing paragraph titles remain valid for paragraph retrieval. The audit retains the question IDs and original anomalous (title, sentence-index) pairs, with counts by split. This experiment does not assess sentence-level support. MuSiQue uses its original paragraph text and integer paragraph indices. Its per-paragraph support flags must agree with the decomposition's support indices. The decomposition never enters a model or graph.

Canonical documents have exactly id, title and text. Each ID hashes compact UTF-8 JSON [title,text]. Exact duplicates collapse, while different bodies under the same title retain distinct IDs. Every source index maps to its canonical ID. The isolated judgment records the original positive paragraph count and unique relevant IDs. No row lacking positive support can enter scoring. Each paragraph retains its source article title for attribution.

## Model and six rankers

The sparse baseline is the engine's fixed BM25 configuration. MiniLM uses the Apache-2.0 all-MiniLM-L6-v2 revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Its six allowlisted files and independent SHA-256 values are in `benchmarks/model.json`. Each loader sees a newly created directory of individually verified files. It uses offline local-only loading, safetensors, no remote code, evaluation mode, and CPU float32 SDPA with six intra-op threads, one inter-op thread, deterministic algorithms, and a non-parallel tokenizer. Model files stay in the private download cache. [Pinned upstream model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41)

Document encoder input is the original title, newline and original body. Query input is the original question. Tokenization includes special tokens and caps the sequence at 256. Each vector is the L2-normalized attention-mask mean of final hidden states, including unmasked special tokens. The 384 coordinates are retained as float32. Cosine is the float32 sum of float32 elementwise products. Inputs have namespace-qualified keys per dataset and split. A SHA-256 of the encoder settings and allowlisted file hashes identifies the encoding protocol. No embeddings are reused across experimental runs. Per-input original lengths and split-level truncation counts remain available. Small floating-point differences can occur across machines.

For batched evidence, inputs of the same split are sorted by character length then key, chunked in groups of 512, sorted within each group by capped token length then original position, and processed in batches of 32. Timing uses the same weights and encoding implementation, encoding the individual question during every dense-bearing query measurement.

1. BM25 and dense rank scores descending, then paragraph ID ascending. Zero sparse scores and negative cosine scores participate.
2. Hybrid combines the complete rankings, using one-based rank and score (1 − w)/(c + sparse rank) + w/(c + dense rank). The training grid enumerates w in [0.25, 0.50, 0.75] then c in [10, 60], with c changing fastest.
3. Graph-2 uses the hybrid scores as the base for the engine's provenance-backed paragraph walk. After choosing hybrid, train at depth 2 over mix [0.15, 0.35, 0.60] then seeds [1, 2, 3], with seeds changing fastest. Graph-1 and graph-3 use the same hybrid, mix, and seed count. Only traversal depth changes. Graph edges mean exact title mentions or shared title ownership. The engine contract defines all transitions and fallback branches.

Each stage chooses the largest mean AllSupport@4, then largest mean binary nDCG@4, then earliest enumerated setting. Each dataset has independent settings. Both frozen configurations are written before any development embedding or score is computed. All 15 grid cells per training question are retained. Development never enters selection.

## Metrics and uncertainty

For cutoffs 2, 4 and 6, AllSupport is one iff all unique relevant paragraphs appear in the prefix. Recall divides the number of prefix hits by the number of unique relevant IDs. Binary nDCG@4 discounts a hit at one-based rank r by 1/log2(r + 1) and divides by the ideal top-four DCG for that question. MRR is the inverse rank of the first hit. Means weight questions equally. Hotpot bridge/comparison and MuSiQue 2/3/4-hop strata are descriptive.

Independent checks evaluate nDCG@4, Recall@2/4/6, and reciprocal rank using the trec_eval C implementation through pytrec_eval-terrier. It receives positive, strictly descending integer rank-surrogate scores for every ranked candidate so trec_eval cannot re-break ties or drop zero-scored candidates. The check reconstructs the surrogate order before calling the evaluator. A separate set-subset check assesses AllSupport at all three cutoffs. The run records per-metric counts and worst absolute errors. Absolute error above 1e-10 fails the run.

The prespecified primary contrast is graph-2 minus hybrid on AllSupport@4 in each development set. Report wins, losses, ties, mean paired delta, two-sided exact binomial sign-flip p conditional on discordance (p = 1 for no discordance), and Holm-adjusted p across the two datasets. Query-paired percentile intervals resample questions with replacement 10,000 times using NumPy PCG64 seed 20261119 and linear 2.5/97.5 percentiles.

The supplementary cluster analysis connects any two questions sharing a relevant paragraph ID and takes transitive components. Each draw resamples as many components as observed, uniformly with replacement. Its statistic is the total sampled delta divided by the total sampled question count. The component bootstrap restarts the same seed and uses the same draw count and percentiles. Report component count, sorted sizes and largest share. Fewer than two components yields unavailable cluster uncertainty. Shared questions, titles with differing text and shared subquestions can create further dependence.

## Timing and reproduction

Within each development set, take the first 64 IDs under SHA-256 of `graphmind-v1-timing:` plus ID, ties by ID. Build one graph and encode its candidate document matrix once per case. Record those setup durations separately. For each case run six sequential passes over methods in the displayed order, recording the first as warmup and the remaining five as observations. Every BM25-bearing timed query includes document lexical tokenization, BM25 scoring and the required ranking. Each dense-bearing query includes question tokenization, encoding and dot products. Hybrid also performs full fusion and sorting. Each graph method includes fusion plus its own complete graph expansion and rank extraction. The timers use perf_counter_ns. Save all samples and report pooled p50/p95 across 320 warm measurements per method per dataset. Model loading, offline bulk encoding, corpus reads and graph construction are outside warm query times. Hardware and software versions accompany the results. These local sequential timings do not estimate production throughput.

From the checkout root in a Python 3.12+ virtual environment:

```sh
python -m pip install -e . -r requirements-benchmark.txt
python -m scripts.download_data --cache data/cache
python -m benchmarks.run_retrieval --cache data/cache --output results/retrieval
```

Choose a new output directory for a reproduction. The runner stages hashed shards in a sibling temporary directory and publishes the complete directory only after every measurement and check succeeds. A failed run leaves its partial observations for inspection. Compressed JSONL case, judgment, run, grid and metric tables use up to 64 records per shard. Vectors occupy little-endian row-major float32 gzip shards with adjacent key/token-length arrays. Public training text is limited to the selected 500 questions per dataset. The retrieval manifest covers all files in the completed directory except itself. A digest authenticates artifact bytes relative to the manifest and cannot establish the origin of a timing or prove that any model executed.
