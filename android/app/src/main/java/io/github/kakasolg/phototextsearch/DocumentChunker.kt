package io.github.kakasolg.phototextsearch

/**
 * Splits a document into fixed-length chunks for a short-sequence embedding model (seq64), so a long document
 * is not silently cut to its first 64 tokens. Pure token arithmetic; tokenization happens in the caller.
 *
 * Every chunk is `BOS + "title: {title} | text:" + body slice + EOS` and fits in [seqLen].
 *
 * ── 왜 프리픽스와 본문을 따로 토큰화하는가 (2026-10-14 확인) ─────────────
 *  · Gemma v1 토크나이저는 공백을 "▁"로 바꿀 뿐 단어 단위로 미리 자르지 않아서, 문자열을 어디서 끊느냐에 따라
 *    토큰이 달라진다("text: hello"는 `▁hello` 하나, 따로 자르면 `▁` + `hello`).
 *  · 그래서 프리픽스는 끝 공백 없이 `"... | text:"`로, 본문은 앞에 공백을 붙인 `" " + body`로 토큰화한다.
 *    공개 벤치 문서 2,000건에서 이 둘을 이은 것이 문서 전체를 한 번에 토큰화한 결과와 **전부 같았다** —
 *    그래서 한 청크에 들어가는 문서는 청킹 전과 똑같은 토큰(=똑같은 벡터)을 받는다.
 *  · 두 번째 청크부터는 본문 조각이 단어 중간에서 시작할 수 있다. 단어 경계로 맞추려면 어휘 문자열이
 *    필요한데 지금 토크나이저 블롭엔 없어서 고정 폭으로 자른다.
 *
 * ── 제목이 너무 길 때 ──────────────────────────
 *  본문 자리가 [MIN_BODY_TOKENS]보다 작아지면(공개 문서 중 제목만으로 59토큰인 것이 있었다) 제목을
 *  프리픽스에서 빼고 본문 앞에 붙인다: `title: none | text: {title} {body}`. 제목 내용은 첫 청크에 남는다.
 */
object DocumentChunker {

    const val MIN_BODY_TOKENS = 16

    /** Prefix text without the trailing space; the space belongs to the body (see class comment). */
    fun prefixFor(title: String?): String = "title: ${title?.takeIf { it.isNotBlank() } ?: "none"} | text:"

    class Plan(
        /** Token IDs per chunk, each ≤ seqLen and ending with EOS. */
        val chunks: List<IntArray>,
        /** Body tokens available per chunk. */
        val bodyBudget: Int,
        /** True when the title was moved into the body because it left too little room. */
        val titleMovedToBody: Boolean,
    )

    /**
     * @param prefixIds `encode(prefixFor(title))` with special tokens — starts with BOS, ends with EOS (dropped here).
     * @param bodyIds `encode(" " + body, addSpecialTokens = false)`.
     */
    fun plan(prefixIds: IntArray, bodyIds: IntArray, eosId: Int, seqLen: Int): Plan? {
        val head = if (prefixIds.isNotEmpty() && prefixIds.last() == eosId) prefixIds.copyOf(prefixIds.size - 1) else prefixIds
        val budget = seqLen - head.size - 1
        if (budget < MIN_BODY_TOKENS) return null
        if (bodyIds.size <= budget) return Plan(listOf(head + bodyIds + eosId), budget, titleMovedToBody = false)
        val chunks = ArrayList<IntArray>((bodyIds.size + budget - 1) / budget)
        var start = 0
        while (start < bodyIds.size) {
            val end = minOf(start + budget, bodyIds.size)
            chunks += head + bodyIds.copyOfRange(start, end) + eosId
            start = end
        }
        return Plan(chunks, budget, titleMovedToBody = false)
    }

    /** Tokenizes and plans; falls back to moving the title into the body when the title is too long. */
    fun plan(tokenizer: GemmaBpeTokenizer, title: String?, body: String, seqLen: Int): Plan {
        plan(tokenizer.encode(prefixFor(title)), tokenizer.encode(" $body", addSpecialTokens = false), tokenizer.eosId, seqLen)
            ?.let { return it }
        val moved = plan(
            tokenizer.encode(prefixFor(null)),
            tokenizer.encode(" ${title.orEmpty()} $body", addSpecialTokens = false),
            tokenizer.eosId,
            seqLen,
        )
        return checkNotNull(moved) { "seqLen $seqLen is too short for the bare prefix" }
            .let { Plan(it.chunks, it.bodyBudget, titleMovedToBody = true) }
    }

    /** Document score = best chunk. Vectors are unit length, so the dot product is the cosine. */
    fun maxScore(query: FloatArray, chunkVectors: List<FloatArray>): Float =
        chunkVectors.maxOf { EmbeddingGemmaEmbedder.cosine(query, it) }
}
