"""Build gemma_tokenizer.bin, the compact tokenizer the Android demo (GemmaBpeTokenizer.kt) reads.

Parsing the 33 MB tokenizer.json on a phone is slow and memory-hungry, and encoding does not need the string vocabulary:
both sides of every BPE merge are vocabulary tokens, so after one conversion to integer IDs the merge loop is pure integer
work. The blob holds only what encoding needs: single-character pieces → ID, the 256 byte-fallback IDs, the merges in rank
order, and BOS/EOS. It is a derivative of the Gemma tokenizer, so it is not distributed here — build it yourself.

Format (all little-endian int32):
    magic 'GBPE', version 1, bosId, eosId,
    charCount, then charCount × (codepoint, id),
    256 × byte-fallback id (<0x00> … <0xFF>),
    mergeCount, then mergeCount × (idA, idB, mergedId)      ← order is the rank

Usage:
    python scripts/build_tokenizer_blob.py [--tokenizer path/to/tokenizer.json] [--out gemma_tokenizer.bin]
Without --tokenizer, tokenizer.json is downloaded from google/embeddinggemma-300m (accept the Gemma terms on Hugging Face
first and log in with `huggingface-cli login`).
"""

import argparse
import json
import struct
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=Path)
    ap.add_argument("--out", type=Path, default=Path("gemma_tokenizer.bin"))
    a = ap.parse_args()
    path = a.tokenizer
    if path is None:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download("google/embeddinggemma-300m", "tokenizer.json"))

    d = json.loads(path.read_text(encoding="utf-8"))
    m = d["model"]
    vocab: dict[str, int] = m["vocab"]
    merges = m["merges"]
    assert m.get("byte_fallback"), "the blob format assumes byte_fallback is on"

    chars = {t: i for t, i in vocab.items() if len(t) == 1}
    byte_ids = [vocab.get(f"<0x{b:02X}>", -1) for b in range(256)]
    missing = [b for b, i in enumerate(byte_ids) if i < 0]
    assert not missing, f"missing byte-fallback tokens: {missing[:8]}"

    rows = []
    for pair in merges:
        x, y = pair if isinstance(pair, list) else pair.split(" ", 1)
        ix, iy, im = vocab.get(x), vocab.get(y), vocab.get(x + y)
        # Fail here rather than silently produce wrong tokens on the phone.
        assert ix is not None and iy is not None, f"merge piece not in vocab: {x!r} {y!r}"
        assert im is not None, f"merge result not in vocab: {(x + y)!r}"
        rows.append((ix, iy, im))

    buf = bytearray(b"GBPE")
    buf += struct.pack("<i", 1)
    buf += struct.pack("<ii", vocab["<bos>"], vocab["<eos>"])
    buf += struct.pack("<i", len(chars))
    for t, i in sorted(chars.items(), key=lambda kv: ord(kv[0])):
        buf += struct.pack("<ii", ord(t), i)
    buf += struct.pack("<256i", *byte_ids)
    buf += struct.pack("<i", len(rows))
    for ix, iy, im in rows:
        buf += struct.pack("<iii", ix, iy, im)

    a.out.write_bytes(buf)
    print(f"wrote {a.out}: chars {len(chars)}, merges {len(rows)}, {len(buf) / 1e6:.1f} MB, "
          f"bos {vocab['<bos>']}, eos {vocab['<eos>']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
