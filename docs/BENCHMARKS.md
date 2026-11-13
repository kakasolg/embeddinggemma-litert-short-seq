# Benchmarks: methodology and full results

This page has every number behind the summary tables in the [README](../README.md). Raw results:
[`results/public_bench_pc.json`](../results/public_bench_pc.json),
[`results/device_public_bench.json`](../results/device_public_bench.json),
[`results/device_bench_sample.json`](../results/device_bench_sample.json).

Models compared:

| Name | What it is |
|---|---|
| Original, first N tokens (`ptN`) | `google/embeddinggemma-300m` in sentence-transformers, fed the same token IDs cut to the first N |
| Original, full (`ptfull`) | the same, up to 2048 tokens |
| seq64 / seq128 / seq256 q8 | this conversion (litert-torch 0.9.4, `dynamic_int8`); only seq64 and seq256 are released |
| int4 block32 / block128 | the same conversion with `dynamic_int4_block32/128`; evaluated, **not released** |
| litert-community seq256 | `embeddinggemma-300M_seq256_mixed-precision.tflite` from `litert-community/embeddinggemma-300m`, for reference |

## Setup

Tasks (loaded with MTEB 2.24.14). Token lengths include the task prompt, BOS and EOS.

| Task | Language | Size | Metric | Token length |
|---|---|---|---|---|
| KorSTS test | Korean | 1,376 pairs | Spearman | sentences: median 23, max 70 (0.4% over 64) |
| STSBenchmark test | English | 1,379 pairs | Spearman | sentences: median 19, max 64 |
| Ko-StrategyQA dev | Korean | 592 queries, 9,251 docs | nDCG@10 | docs: median 166 (94.5% over 64, 19.4% over 256) |
| SciFact test | English | 300 queries, 5,183 docs | nDCG@10 | docs: median 288 (100% over 64, 61.1% over 256) |

Prompts are those in the EmbeddingGemma model card:
- STS: `task: sentence similarity | query: `
- queries: `task: search result | query: `
- documents: `title: {title|none} | text: `

All models receive **identical token IDs**. Inputs that are too long are cut to the first N tokens (EOS dropped),
which is what a fixed-length `.tflite` does. Comparing a tflite file with the original at the same N therefore
isolates conversion and quantization error.

The metrics are computed directly in `scripts/eval_public_bench.py`, not through the MTEB runner. Numbers are
comparable within this page, not with the MTEB leaderboard.

## Quality

| Model | Size | KorSTS | STSB | Ko-StrategyQA | SciFact |
|---|---|---|---|---|---|
| Original, full | — | 0.8091 | 0.8818 | 0.8068 | 0.7921 |
| Original, first 256 | — | 0.8091 | 0.8818 | 0.8056 | 0.7724 |
| Original, first 128 | — | 0.8091 | 0.8818 | 0.7954 | 0.7454 |
| Original, first 64 | — | 0.8091 | 0.8818 | 0.7713 | 0.6795 |
| **seq256 q8** | 314 MB | 0.8086 | 0.8817 | 0.8066 | 0.7758 |
| seq128 q8 (not released) | 314 MB | 0.8086 | 0.8817 | 0.7953 | 0.7422 |
| **seq64 q8** | 314 MB | 0.8086 | 0.8817 | 0.7669 | 0.6791 |
| int4 block32, seq64 (not released) | 175 MB | 0.7942 | 0.8765 | 0.7618 | 0.6567 |
| int4 block128, seq64 (not released) | 161 MB | 0.7911 | 0.8758 | 0.7487 | 0.6632 |
| litert-community seq256 (reference) | 179 MB | 0.8005 | 0.8806 | 0.7994 | 0.7597 |

### Difference from the original at the same length

Paired bootstrap with 1,000 resamples of pairs (STS) or queries (retrieval). Bold means the 95% interval excludes 0.

| Model − original | KorSTS | STSB | Ko-StrategyQA | SciFact |
|---|---|---|---|---|
| seq256 q8 − first 256 | −0.0005 [−0.002, +0.000] | −0.0001 [−0.001, +0.000] | +0.0010 [−0.002, +0.004] | +0.0034 [−0.000, +0.008] |
| seq128 q8 − first 128 | −0.0005 [−0.002, +0.000] | −0.0001 [−0.001, +0.000] | −0.0001 [−0.003, +0.003] | −0.0033 [−0.008, +0.002] |
| seq64 q8 − first 64 | −0.0005 [−0.001, +0.000] | −0.0001 [−0.001, +0.000] | **−0.0044** [−0.008, −0.002] | −0.0003 [−0.006, +0.005] |
| int4 block32 − first 64 | **−0.0150** | **−0.0053** | **−0.0095** | **−0.0228** |
| int4 block128 − first 64 | **−0.0180** | **−0.0059** | **−0.0226** | **−0.0162** |
| litert-community seq256 − first 256 | **−0.0086** [−0.013, −0.004] | −0.0012 [−0.004, +0.001] | −0.0062 [−0.013, +0.000] | **−0.0127** [−0.022, −0.003] |

### Cosine similarity to the original (all 20,836 texts, same truncation length)

| Model | Mean | Min | 1st percentile (texts ≤ 64 tokens) |
|---|---|---|---|
| q8, all lengths | 0.9984–0.9985 | 0.9934 | 0.9976 |
| int4 block32 | 0.9585 | 0.8407 | 0.9331 |
| int4 block128 | 0.9372 | 0.7226 | 0.9003 |
| litert-community seq256 | 0.9636 | 0.7916 | 0.9231 |

For texts of 64 tokens or fewer, the seq64, seq128 and seq256 q8 files return identical vectors (maximum absolute
difference 0.0 over 6,899 texts).

How to read this:
- The q8 conversion cannot be told apart from the original on short text. The one interval that excludes 0 is
  seq64 on Ko-StrategyQA (−0.004), where nearly every document is truncated.
- int4 is measurably worse on every task, so it was not released.
- The litert-community file is a different trade-off (smaller, mixed precision). On this setup it sits slightly
  below the original on KorSTS and SciFact. The difference is small.

## Truncation and chunking

nDCG@10 with the original model, so this measures truncation only:

| Documents embedded as | Ko-StrategyQA | SciFact |
|---|---|---|
| Full length | 0.8068 | 0.7921 |
| First 256 | 0.8056 | 0.7724 |
| First 128 | 0.7954 | 0.7454 |
| First 64 | 0.7713 | 0.6795 |
| 64-token chunks, score = best chunk | 0.7953 | 0.7635 |
| 128-token chunks, score = best chunk | 0.7990 | 0.7776 |

Chunks repeat the `title: … | text: ` prefix and do not overlap. With 64-token chunks the corpora produce 81,699
chunks for 14,434 documents (about 5.7 per document). Five documents whose title alone exceeds 64 tokens produce no
chunk and are scored as missing (`python scripts/eval_public_bench.py chunk 64`).

## On-device

**Device:** Pixel 11 Pro XL (Tensor G6), LiteRT `Interpreter` from Kotlin, CPU with 4 threads requested, phone
connected over USB.

> **Measured later (2026-10-15):** on LiteRT 2.1.5 the classic `Interpreter` does not pass the thread count to its
> XNNPACK delegate, so it runs on one core whatever is requested (a thread monitor on a Galaxy Z Fold4 saw ~0.9 cores busy). `CompiledModel`
> with `CpuOptions(4 threads)` produced bit-identical vectors and was 1.3–2.8× faster on a Pixel 11 Pro XL and a Galaxy
> Z Fold4. The table below is left as measured; for an app, use `CompiledModel` (the demo's `EmbeddingGemmaEmbedder.kt`
> does, with `Interpreter` as a fallback). Absolute Pixel numbers also depend on the CPU group the app runs in — see the
> README's "Two measurement traps".

**Sample:** 300 texts drawn with seed 20261014 by `scripts/device_public_bench.py prep`:
- KorSTS and STSBenchmark: 50 pairs each.
- Ko-StrategyQA and SciFact: 25 queries and 25 documents each.

The median length is 25 tokens; 50 texts are over 64 tokens and 25 are over 256.

**Protocol:** each run embeds all 300 texts once, timing each call with `System.nanoTime`, and the first 3 items
are dropped as warm-up. Thermal status was recorded before and after each run, without waiting for the phone to
cool. Fourteen runs went back to back in alternating order. CPU frequency caps (`scaling_max_freq`) and the
skin-temperature estimate were logged around the last four runs.

| Position | File | Mean ms | p95 ms | Thermal status | Big-core cap before → after |
|---|---|---|---|---|---|
| 1 | seq64 q8 | 29.8 | 30.6 | 0 → 0 | |
| 2 | seq256 q8 | 147.8 | 151.9 | 0 → 0 | |
| 3 | litert-community seq256 | 147.1 | 151.9 | 0 → 0 | |
| 4 | seq64 q8 | 32.6 | 33.2 | 0 → 0 | |
| 5 | seq256 q8 | 162.8 | 168.1 | 0 → 0 | |
| 6 | litert-community seq256 | 192.3 | 238.6 | 0 → 0 | |
| 7 | int4 block32 seq64 | 121.2 | 131.8 | 0 → 0 | |
| 8 | litert-community seq256 | 196.2 | 238.2 | 0 → 0 | |
| 9 | seq256 q8 | 243.5 | 343.7 | 0 → 0 | |
| 10 | seq64 q8 | 61.4 | 66.7 | 0 → 0 | (4,109 MHz max; 2,496 MHz measured after run 10) |
| 11 | seq64 q8 | 36.8 | 37.8 | 0 → 0 | 2,803 → 2,496 MHz |
| 12 | litert-community seq256 | 208.5 | 338.4 | 0 → 1 | 2,496 → 2,150 MHz |
| 13 | seq256 q8 | 234.9 | 236.5 | 1 → 1 | 1,996 → 1,843 MHz |
| 14 | seq64 q8 | 71.1 | 86.4 | 1 → 1 | 1,843 → 2,150 MHz |

Cosine of the device vectors to the original, at the same truncation length:

| File | Mean | Min | 1st percentile |
|---|---|---|---|
| seq64 q8 | 0.9986 | 0.9971 | 0.9975 |
| seq256 q8 | 0.9986 | 0.9971 | 0.9976 |
| int4 block32 seq64 | 0.9666 | 0.9141 | 0.9335 |
| litert-community seq256 | 0.9643 | 0.9008 | 0.9326 |

Device vectors were identical across runs of the same file.

How to read this:
- Under the same conditions (runs 1–3), seq64 takes about 1/5 of the time of seq256. The released seq256 file and
  the litert-community seq256 file run at the same speed, so the speedup comes from the shorter length, not from
  the conversion.
- Sustained load lowers the CPU frequency cap long before the thermal status changes. The big core's maximum is
  4,109 MHz; by runs 11–14 its cap was 1.8–2.8 GHz, while the status stayed at 0 until after run 12. Report
  cold-start and sustained numbers separately.
- int4 (run 7, measured after the cap had started to drop) is slower than seq64 q8 measured around it (32.6 ms in
  run 4, 61.4 ms in run 10).

### Running the device benchmark yourself

`scripts/device_public_bench.py prep` writes the token IDs to `$EGSS_CACHE/device_bench_sample.ids`. The format is
one line per text with comma-separated IDs. Any runner works if it writes the formats that `score` reads (see the
script's docstring). The core of the Kotlin runner used here:

```kotlin
val interpreter = Interpreter(modelFile, Interpreter.Options().setNumThreads(4))
val seq = interpreter.getInputTensor(0).shape()[1]
val out = ByteBuffer.allocateDirect(768 * 4).order(ByteOrder.nativeOrder())
for (ids in rows) {                                   // rows: List<IntArray> from the .ids file
    val t0 = System.nanoTime()                        // timed: buffer fill + run + read-out
    val used = minOf(ids.size, seq)                   // head truncation, like the PC scripts
    val idBuf = ByteBuffer.allocateDirect(seq * 8).order(ByteOrder.nativeOrder())
    val maskBuf = ByteBuffer.allocateDirect(seq * 8).order(ByteOrder.nativeOrder())
    for (i in 0 until seq) {
        idBuf.putLong(if (i < used) ids[i].toLong() else 0L)
        maskBuf.putLong(if (i < used) 1L else 0L)
    }
    idBuf.rewind(); maskBuf.rewind(); out.rewind()
    interpreter.runForMultipleInputsOutputs(arrayOf(idBuf, maskBuf), mapOf(0 to out))
    out.rewind()
    val vector = FloatArray(768) { out.float }        // already L2-normalized -> <name>.f32
    val ms = (System.nanoTime() - t0) / 1e6           // append to <name>.ms.txt
}
```

The timing covers the whole embed call: filling the input buffers, running the model and reading out the vector. Model loading is excluded.

## Caveats

- The tables in this document are from one device (Pixel 11 Pro XL) on the CPU with `Interpreter`. The later
  `CompiledModel` check (Pixel 11 Pro XL and Galaxy Z Fold4) and the GPU results (Galaxy Z Fold4: the q8 files do not
  compile; use the CPU) are in the README's [Usage](../README.md#usage) section. NPU delegates were not measured.
- PC CPU timings are not reported, because the models ran concurrently during the PC evaluation.
- STS and retrieval scores come from four tasks in two languages. They are meant to show conversion fidelity,
  not general model quality.
