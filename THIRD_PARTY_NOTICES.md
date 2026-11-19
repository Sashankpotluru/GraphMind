# Third-party material

The MIT license covers the original GraphMind code. Dataset and model terms follow their upstream releases.

**HotpotQA.** Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov and Christopher D. Manning created HotpotQA (2018). The supplied distractor contexts contain Wikipedia article text credited to the article contributors. The dataset and the adapted Hotpot cases remain available under CC BY-SA 4.0. GraphMind concatenates supplied sentences, deduplicates exact title/body pairs, attaches content IDs and extracts lexical links. The original article title in each document or source judgment identifies its article on English Wikipedia and its contribution history. [Dataset and paper](https://hotpotqa.github.io/) · [Official data distribution](https://huggingface.co/datasets/hotpotqa/hotpot_qa) · [License](https://creativecommons.org/licenses/by-sa/4.0/)

**MuSiQue.** Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot and Ashish Sabharwal created MuSiQue (2022). The authors distribute MuSiQue under CC BY 4.0. The source draws on SQuAD, T-REx, Natural Questions, MLQA and Zero Shot RE. Each candidate's source title identifies its Wikipedia article and contributors. GraphMind retains the supplied title and body, maps paragraph support to content IDs, deduplicates exact pairs and derives lexical links. The adapted MuSiQue material retains its dataset attribution and CC BY 4.0 terms. [Authors' repository, source and usage notes](https://github.com/StonyBrookNLP/musique) · [License](https://creativecommons.org/licenses/by/4.0/)

**MiniLM.** The sentence-transformers all-MiniLM-L6-v2 release is licensed under Apache 2.0. The benchmark uses its verified tokenizer/configuration and safetensors weights locally. The repository publishes derived numeric representations and retrieval outputs. [Pinned model and card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41) · [License](https://www.apache.org/licenses/LICENSE-2.0)

**Evaluation software.** The independent metric comparison uses pytrec_eval-terrier, a Python interface to NIST trec_eval. Reproduction installs the package separately. [Python interface](https://github.com/terrierteam/pytrec_eval) · [NIST trec_eval](https://github.com/usnistgov/trec_eval)

**FLAN-T5-small.** The matched reader uses Google's Apache-2.0 FLAN-T5-small release. The unmodified tokenizer JSON is redistributed for weight-free decoding and replay. The weights are downloaded separately. [Pinned model and card](https://huggingface.co/google/flan-t5-small/tree/0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab) · [Apache license text](benchmarks/vendor/flan/LICENSE)

**Answer evaluation.** HotpotQA’s authors distribute their scoring code under Apache 2.0. MuSiQue’s authors distribute their scorer under CC BY 4.0 and credit AllenNLP’s SQuAD tools. The vendored Hotpot module uses the standard JSON import. The MuSiQue answer module uses a relative Metric import. Their answer functions are otherwise unchanged. Pinned source URLs and original SHA-256 values accompany each module. [Source entries](benchmarks/vendor/scorers.json) · [Hotpot license](benchmarks/vendor/hotpot/LICENSE.txt) · [MuSiQue license](benchmarks/vendor/musique/LICENSE)

**Browser runtime.** The demo serves five unmodified files from Pyodide 0.29.3 (CPython 3.13.2) under the same origin as the workspace. The distribution archive is 5,598,203 bytes, SHA-256 `924f3167d6bf5e553bda392615eaa82edd82a9cffc32a1cc49853a926b148c5c`. The selected runtime files total 12,283,621 bytes. The lockfile is upstream loader metadata. Compatible optional packages are listed there, and the deployed worker uses only the standard library. The build downloads the archive into a local cache and unpacks the five listed files into the generated site. [Pinned distribution archive](https://registry.npmjs.org/pyodide/-/pyodide-0.29.3.tgz) · [Pyodide release and corresponding source](https://github.com/pyodide/pyodide/tree/0.29.3) · [Pyodide MPL 2.0](web/assets/Pyodide-LICENSE.txt) · [Python license and acknowledgments](web/assets/Python-LICENSE.txt)

| Runtime file | Bytes | SHA-256 |
| --- | ---: | --- |
| pyodide.mjs | 17,616 | `23213d27e9dacbda6f25f955fb38dabfc40ada082c04ee253047ea5dddcaa3b5` |
| pyodide.asm.js | 1,074,322 | `1263f02b5b26099b96112378156f242dd98b39a8201ba7765e5fe3d455c5ce91` |
| pyodide.asm.wasm | 8,647,684 | `e2f4ee75b325e35eb31bfb8c613d4dd5098f5502c156a97847686875b5025480` |
| python_stdlib.zip | 2,423,989 | `4298b6ee445cb724c3973437da47789752b9e6ff4e26619026b283ec801fc46b` |
| pyodide-lock.json | 120,010 | `3256ffc76388de0e37f4b34d42ab484268d1afc675179ff97b2a5bb14f84ccac` |

**Typography.** The Bricolage Grotesque Project Authors created the typeface used in GraphMind’s interface. The Latin variable-weight subset comes from Fontsource 5.3.0. [Typeface source](https://github.com/ateliertriay/bricolage) · [Fontsource distribution](https://registry.npmjs.org/@fontsource-variable/bricolage-grotesque/-/bricolage-grotesque-5.3.0.tgz) · [SIL Open Font License](web/assets/Font-LICENSE.txt)
