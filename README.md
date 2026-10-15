# EmbeddingGemma-300M short-sequence LiteRT builds (seq64 / seq256, int8)

LiteRT (`.tflite`) conversions of [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)
with **fixed input lengths of 64 and 256 tokens**, built for on-device use where most inputs are short
(search queries, sentences, short notes).

- **Close to the original model.** Mean cosine similarity to the original PyTorch embeddings is **0.9985**
  (minimum 0.9934 over 20,836 public-benchmark texts, compared at the same truncation length).
- **Fast on a phone for short text.** The seq64 file takes **~30 ms per text** on a Pixel 11 Pro XL CPU (4 threads).
  That is about 5× faster than a seq256 file on the same device under the same conditions.
- **Size.** About 314 MB per file (dynamic int8).

> **These are modified files, not Google's originals.** They are Model Derivatives of `google/embeddinggemma-300m`,
> produced by converting the original checkpoint with
> [litert-torch](https://github.com/google-ai-edge/litert-torch) (`dynamic_int8` quantization, fixed sequence
> lengths). No weights were retrained or fine-tuned. See [Conversion](#conversion).

## License and use restrictions

Gemma is provided under and subject to the **Gemma Terms of Use** found at
[ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms). By downloading or using the model files you agree
to those terms. The required notice is in [`NOTICE`](NOTICE).

Use of the model files is subject to the use restrictions in Section 3.2 of the Gemma Terms of Use, including the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy). These restrictions apply to you
and to anyone you redistribute the files (or further derivatives) to. If you redistribute them, you must pass
these restrictions on and include the Gemma Terms of Use and the `NOTICE` file.

Any license that applies to this repository's own scripts and documentation does not apply to the model files.
The model files are governed by the Gemma Terms of Use.

## Download

The model files are **not stored in git**. Download them from the
[GitHub Releases](https://github.com/kakasolg/embeddinggemma-litert-short-seq/releases) page.

| File | Input length | Size (bytes) | SHA256 | Use for |
|---|---|---|---|---|
| `embeddinggemma-300m_seq64_q8.tflite` | 64 tokens | 314,071,856 | `289ec1bcd3b3122cf91bfacd2eba98980a769d31665b80c8544434217660964e` | queries, sentences, short notes |
| `embeddinggemma-300m_seq256_q8.tflite` | 256 tokens | 314,497,248 | `eadc04a177b045c354185af18a301d08477382d7c43e9d9b8db3e0319ab7c729` | longer passages |

```bash
curl -LO https://github.com/kakasolg/embeddinggemma-litert-short-seq/releases/download/v1.0.0/embeddinggemma-300m_seq64_q8.tflite
```

```bash
curl -LO https://github.com/kakasolg/embeddinggemma-litert-short-seq/releases/download/v1.0.0/embeddinggemma-300m_seq256_q8.tflite
```

Or download both with the GitHub CLI:

```bash
gh release download v1.0.0 --repo kakasolg/embeddinggemma-litert-short-seq --pattern "*.tflite" --pattern SHA256SUMS
```

Verify the files:

```bash
sha256sum -c SHA256SUMS
```

Both files come from the same checkpoint with the same recipe. For inputs that fit in 64 tokens they return
**identical** vectors (checked on 6,899 benchmark texts), so you can mix them in one index. For example, use seq64
for queries and seq256 for documents. Do **not** mix them with vectors from other conversions of EmbeddingGemma.
Those are not interchangeable (cosine around 0.96 against these files).

## Usage

**Signature** (`serving_default`):

| Tensor | Shape | Type | Content |
|---|---|---|---|
| `args_0` | `[1, L]` | int64 | token IDs, zero-padded |
| `args_1` | `[1, L]` | int64 | attention mask (1 for real tokens, 0 for padding) |
| `output_0` | `[1, 768]` | float32 | sentence embedding, already pooled, projected and L2-normalized |

The input names carry no meaning. Bind by position: the first input is the token IDs and the second is the mask.

Tokenize with the original tokenizer (`google/embeddinggemma-300m`, with BOS/EOS) and use EmbeddingGemma's
task prompts:

| Task | Prefix |
|---|---|
| Search query | `task: search result \| query: ` |
| Document | `title: {title or "none"} \| text: ` |
| Sentence similarity | `task: sentence similarity \| query: ` |

```python
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("google/embeddinggemma-300m")
it = Interpreter(model_path="embeddinggemma-300m_seq64_q8.tflite", num_threads=4)
it.allocate_tensors()
ids_in, mask_in = it.get_input_details()
out = it.get_output_details()[0]
L = ids_in["shape"][-1]

def embed(text: str) -> np.ndarray:
    ids = tok(text)["input_ids"]
    if len(ids) > L:
        print(f"warning: {len(ids)} tokens, truncated to {L}")  # see "Truncation" below
    ids = ids[:L]
    x = np.zeros((1, L), np.int64); x[0, :len(ids)] = ids
    m = np.zeros((1, L), np.int64); m[0, :len(ids)] = 1
    it.set_tensor(ids_in["index"], x)
    it.set_tensor(mask_in["index"], m)
    it.invoke()
    return it.get_tensor(out["index"])[0]

q = embed("task: search result | query: What is Tony Bennett's middle name?")
d = embed("title: none | text: Anthony Dominick Benedetto, known professionally as Tony Bennett ...")
print(float(q @ d))
```

On Android the same file runs with the LiteRT `Interpreter` (CPU, 4 threads in the measurements below).
Tokenize ahead of time or with a Gemma-compatible tokenizer, and fill the two inputs the same way.

## ⚠️ Truncation: longer inputs are cut silently

The input shape is fixed. Text longer than L tokens (counting the task prefix and BOS) keeps only its
**first L tokens**. The rest, including EOS, is dropped. **No error is raised, and the vector looks normal.**
Count tokens before embedding and flag truncation yourself. The Android wrapper used for the measurements exposes
this as a `lastInputWasTruncated` flag.

Rough sizes (Gemma tokenizer, prompt included):
- A short Korean or English sentence is 20–25 tokens.
- A Korean Wikipedia paragraph has a median of 166 tokens.
- A scientific abstract has a median of 288 tokens.
- **64 tokens is about 2–3 short Korean sentences, or about 40 English words.**

Effect on long-document retrieval (nDCG@10 with the original model, so this measures truncation, not conversion):

| How documents are embedded | Ko-StrategyQA | SciFact |
|---|---|---|
| Full length (up to 2048 tokens) | 0.8068 | 0.7921 |
| First 256 tokens | 0.8056 | 0.7724 |
| **First 64 tokens (seq64, plain truncation)** | **0.7713** | **0.6795** |
| **64-token chunks, document score = best chunk** | **0.7953** | **0.7635** |

Short-text tasks lose nothing at 64 tokens: KorSTS and STSBenchmark scores are the same at 64, 256 and full length.

### Recommended use

- **seq64:** queries, single sentences, short notes, titles. At this length it matches the original model and is
  the fastest option.
- **Long documents:** do not feed them to seq64 as-is (SciFact drops from 0.792 to 0.680). Instead, either:
  - split them into 64-token chunks (repeat the `title: … | text: ` prefix on each chunk) and score each document
    by its best chunk. This recovers 0.680 → 0.764 on SciFact and 0.771 → 0.795 on Ko-StrategyQA, at the cost of
    about 5–6 embeddings per document of that length; or
  - embed documents with **seq256** and queries with **seq64**. The vectors are compatible (see [Download](#download)).

## Benchmarks

Full tables, methodology and caveats: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). Raw results are in
[`results/`](results/).

### Quality on public benchmarks (PC, same token IDs for every model)

Scores use the MTEB task definitions (2.24.14 loaders): Spearman for STS and nDCG@10 for retrieval. They use the
prompts above and the stated truncation. They are for comparison within this table, not leaderboard submissions.

| Model | KorSTS | STSBenchmark | Ko-StrategyQA | SciFact |
|---|---|---|---|---|
| Original PyTorch, full length | 0.8091 | 0.8818 | 0.8068 | 0.7921 |
| Original PyTorch, first 256 tokens | 0.8091 | 0.8818 | 0.8056 | 0.7724 |
| **seq256 q8 (this repo)** | 0.8086 | 0.8817 | 0.8066 | 0.7758 |
| Original PyTorch, first 64 tokens | 0.8091 | 0.8818 | 0.7713 | 0.6795 |
| **seq64 q8 (this repo)** | 0.8086 | 0.8817 | 0.7669 | 0.6791 |

Differences from the original at the same length (paired bootstrap, 95% interval):
- STS: within ±0.001 on both tasks for both files.
- Retrieval: seq256 +0.001 (Ko-StrategyQA) / +0.003 (SciFact); seq64 −0.004 / −0.000.

Mean cosine to the original is 0.9985 for both files (minimum 0.9934, 1st percentile 0.9976).

For reference, the existing `litert-community/embeddinggemma-300m` seq256 file (179 MB, mixed precision) scores
0.8005 / 0.8806 / 0.7994 / 0.7597 on the same setup, with mean cosine 0.964 to the original. It is smaller and
supports longer inputs. The files here trade size for closeness to the original and short fixed lengths.

### On-device (Pixel 11 Pro XL, Tensor G6)

The sample is 300 texts drawn with a fixed seed (20261014) from the four tasks above: 100 STS pairs, 50 queries
and 50 documents. Setup: CPU, 4 threads, LiteRT Interpreter, phone connected over USB. The first 3 items of each
run are excluded as warm-up.

| File | Cosine to original (mean / min) | First run, ms per text (p95) | Later runs, ms per text |
|---|---|---|---|
| seq64 q8 | 0.9986 / 0.9971 | **29.8 (30.6)** | 32.6 – 71.1 |
| seq256 q8 | 0.9986 / 0.9971 | 147.8 (151.9) | 162.8 – 243.5 |

Latency depends on the fixed length, not on the text, because every input runs all L positions.

The 14 runs were back to back with no cooldown. Under that sustained load the phone lowered its big-core frequency
cap from 4.1 GHz to 1.8–2.8 GHz, and both files slowed down, while the reported thermal status stayed at 0 for most
of it. seq64 remained 3–4× faster than seq256 (36.8 vs 234.9 ms). If you embed many texts in one go, plan for the
"later runs" column.

## Conversion

```bash
pip install -r requirements-convert.txt
```

```bash
CK=<local snapshot of google/embeddinggemma-300m>
for L in 64 256; do
  python -m litert_torch.generative.examples.embedding_gemma.convert_to_tflite \
    --checkpoint_path=$CK --output_path=out --prefill_seq_lens=$L \
    --quantize=dynamic_int8 --output_name_prefix=embeddinggemma-300m_seq$L
done
```

The converter writes `embeddinggemma-300m_seq{64,256}_q8.tflite`. The converter's int4 options
(`dynamic_int4_block32/128`) give 161–175 MB files. They were evaluated and not released: quality was lower on all
four tasks (−0.005 to −0.023), and they ran slower than int8 on the phone (see `docs/BENCHMARKS.md`).

## Reproducing the benchmarks

```bash
pip install -r requirements-bench.txt
```

```bash
python scripts/eval_public_bench.py embed pt:64 pt:256 pt:full embeddinggemma-300m_seq64_q8.tflite embeddinggemma-300m_seq256_q8.tflite
```

```bash
python scripts/eval_public_bench.py score
```

```bash
python scripts/eval_public_bench.py chunk 64
```

The device steps are described in `docs/BENCHMARKS.md`. The datasets are downloaded through MTEB and are not
redistributed here; only scores and row numbers are stored in `results/`.

## Repository layout

```
README.md                 this file
NOTICE                    Gemma notice (required for redistribution)
SHA256SUMS                checksums of the release assets
docs/BENCHMARKS.md        full results and methodology
scripts/                  benchmark scripts (PC and on-device scoring)
results/                  raw benchmark results (JSON)
requirements-*.txt        pinned environments
```

## Limitations

- Fixed length: inputs over 64 / 256 tokens are truncated silently (see above).
- Each file is about 314 MB; the vocabulary embedding table (262,144 × 768) makes up most of it.
- Measured on one Android device, CPU only. GPU/NPU delegates, iOS and web were not tested.
- The benchmarks cover Korean and English only. EmbeddingGemma itself is multilingual.
- Matryoshka truncation of the output (512/256/128 dimensions) was not evaluated for these files.

## Acknowledgements

EmbeddingGemma is developed by Google DeepMind. See the original
[model card](https://huggingface.co/google/embeddinggemma-300m) and cite the original model when you use these
files. The conversion uses [litert-torch](https://github.com/google-ai-edge/litert-torch) and the benchmarks use
[MTEB](https://github.com/embeddings-benchmark/mteb).
