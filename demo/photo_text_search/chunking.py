"""The PC side of the Android wrapper's chunking (DocumentChunker.kt) and the original-model embedder, for pc_baseline.py.

plan(): same rule as DocumentChunker.plan — the prefix "title: X | text:" is tokenized without its trailing space and the
body as " " + body, each chunk is BOS + prefix + body slice + EOS, at most L tokens. If the title leaves fewer than
MIN_BODY body tokens per chunk, the title moves into the body and the prefix becomes "title: none | text:".
embed_pt(): original PyTorch EmbeddingGemma-300M (sentence-transformers), same token IDs, truncated to n tokens.
"""

import numpy as np

MIN_BODY = 16
PREFIX_QUERY = "task: search result | query: "


def tok_():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("google/embeddinggemma-300m")


def plan(tok, title, body, L=64, eos=True):
    def go(t, b):
        head = tok(f"title: {t or 'none'} | text:")["input_ids"][:-1]
        bids = tok(" " + b, add_special_tokens=False)["input_ids"]
        budget = L - len(head) - (1 if eos else 0)
        if budget < MIN_BODY:
            return None
        tail = [tok.eos_token_id] if eos else []
        if len(bids) <= budget:
            return [head + bids + tail]
        return [head + bids[s:s + budget] + tail for s in range(0, len(bids), budget)]
    return go(title, body) or go(None, f"{title or ''} {body}")


def embed_pt(n, ids):
    """Embed lists of token IDs with the original model; n = max tokens ("full" = 2048). Returns (N, 768) float32."""
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st = SentenceTransformer("google/embeddinggemma-300m", device=dev)
    L = 2048 if n == "full" else int(n)
    V = np.zeros((len(ids), 768), np.float32)
    order = np.argsort([len(x) for x in ids])
    with torch.no_grad():
        for s in range(0, len(order), 32):
            idx = order[s:s + 32]
            xs = [ids[i][:L] for i in idx]
            m = max(len(x) for x in xs)
            inp = torch.zeros((len(xs), m), dtype=torch.long)
            att = torch.zeros_like(inp)
            for k, x in enumerate(xs):
                inp[k, :len(x)] = torch.tensor(x)
                att[k, :len(x)] = 1
            o = st({"input_ids": inp.to(dev), "attention_mask": att.to(dev)})["sentence_embedding"]
            V[idx] = torch.nn.functional.normalize(o.float(), dim=-1).cpu().numpy()
    return V
