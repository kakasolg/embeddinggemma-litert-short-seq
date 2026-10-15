package io.github.kakasolg.phototextsearch

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Gemma 계열 BPE 토크나이저의 온디바이스 구현. EmbeddingGemma가 쓰는 262k 어휘를 그대로 쓴다.
 * 블롭 v2는 Laya(mmBERT, Gemma 2 어휘 256k)용이다 — added token 분리와 Metaspace 분할이 더해진다.
 *
 * ── 왜 직접 구현했나 ────────────────────────────
 * 안드로이드에서 쓸 수 있는 기성품이 없었다. 전부 실제로 확인한 결과다:
 *  · DJL(`ai.djl.huggingface:tokenizers`, `ai.djl.sentencepiece`) — jar에 Android ABI 네이티브가
 *    아예 없다. 데스크톱 전용이다.
 *  · `io.gitlab.shubham0204:sentence-embeddings` — Android ABI는 있는데 Gemma 토크나이저에서
 *    **SIGABRT로 프로세스가 죽는다**(BERT 계열용 AAR이다).
 *  · `litert-support` — 클래스가 0개인 껍데기.
 *  · `tensorflow-lite-task-text` — 토크나이저를 공개 API로 노출하지 않는다.
 *  · `litertlm-android` — LLM 런타임인데 토크나이저를 노출하지 않는다(72개 클래스 중 없음).
 *
 *
 * ── 왜 tokenizer.json을 안 읽고 블롭을 읽나 ──────────────
 * 원본 tokenizer.json은 33MB다. 그런데 **인코딩에는 문자열 vocab이 필요 없다** — merge 쌍의
 * 양쪽은 항상 vocab에 있는 토큰이라, 한 번 정수 ID로 바꿔 두면 병합 루프가 순수 정수 연산이
 * 된다. `scripts/build_tokenizer_blob.py`가 그 정수만 뽑아 6.3MB 블롭으로 굽는다. 디코딩은
 * 이 트랙(임베딩)에서 쓰지 않으므로 문자열 테이블을 아예 담지 않았다 — 디코딩이 필요해지면
 * 블롭 포맷부터 늘려야 한다.
 *
 * 개발 때 공식 HF 토크나이저가 뽑은 ID(정답지)와 비교했고,
 * 파이썬 레퍼런스 구현이 60/60으로 재현하는 걸 먼저 확인한 뒤 이 구현을 썼다(둘 다 이 저장소에는 없음).
 */
class GemmaBpeTokenizer private constructor(
    val bosId: Int,
    val eosId: Int,
    private val charIds: IntArray,
    private val charCodepoints: IntArray,
    private val byteIds: IntArray,
    private val mergeKeys: LongArray,
    private val mergeValues: LongArray,
    private val metaspace: Boolean = false,
    private val added: List<AddedToken> = emptyList(),
) {

    /** 블롭 v2의 added token. 원문에서 먼저 떼어낸다. [lstrip]이면 앞쪽 공백을 같이 먹는다. */
    class AddedToken(val text: String, val id: Int, val lstrip: Boolean)

    /** 첫 글자 → 그 글자로 시작하는 added token(긴 것부터). 매 위치에서 249개를 다 훑지 않으려고. */
    private val addedByFirst: Map<Char, List<AddedToken>> =
        added.sortedByDescending { it.text.length }.groupBy { it.text[0] }

    /**
     * 텍스트를 토큰 ID로. `<bos>` … `<eos>`가 붙는다(tokenizer.json의 TemplateProcessing).
     *
     * 병합 루프는 매 회 전체를 훑어 가장 낮은 rank를 찾는 단순한 O(n²)다. 짧은 메모는 길어야
     * 수십 토큰이라 실측상 문제가 없었다 — 문서 한 편을 통째로 넣게 되면 그때 우선순위 큐로
     * 바꿀 자리다.
     */
    fun encode(text: String, addSpecialTokens: Boolean = true): IntArray {
        val pieces = ArrayList<Int>(text.length + 8)
        if (!metaspace) {
            // v1(EmbeddingGemma): normalizer Replace " " -> "▁"만. Prepend도 분할도 없다.
            bpe(text.replace(' ', '▁'), pieces)
        } else {
            encodeMetaspace(text, pieces)
        }

        if (!addSpecialTokens) return pieces.toIntArray()
        val out = IntArray(pieces.size + 2)
        out[0] = bosId
        for (k in pieces.indices) out[k + 1] = pieces[k]
        out[out.size - 1] = eosId
        return out
    }

    /**
     * v2(Laya/mmBERT): added token을 먼저 떼고, 남은 조각마다 공백→"▁", 맨 앞에 "▁"가 없으면
     * 붙이고(prepend_scheme=always), "▁" 앞에서 잘라 단어별로 BPE를 돌린다. 병합은 단어를 넘지 않는다.
     * 파이썬 레퍼런스 구현과 같은 규칙이다.
     */
    private fun encodeMetaspace(text: String, out: MutableList<Int>) {
        val seg = StringBuilder()
        fun flush() {
            if (seg.isEmpty()) return
            val s = seg.toString().replace(' ', '▁').let { if (it.startsWith('▁')) it else "▁$it" }
            var start = 0
            for (k in 1 until s.length) {
                if (s[k] == '▁') {
                    bpe(s.substring(start, k), out)
                    start = k
                }
            }
            bpe(s.substring(start), out)
            seg.setLength(0)
        }
        var i = 0
        while (i < text.length) {
            val hit = addedByFirst[text[i]]?.firstOrNull { text.startsWith(it.text, i) }
            if (hit == null) {
                seg.append(text[i])
                i++
                continue
            }
            if (hit.lstrip) {
                var end = seg.length
                while (end > 0 && seg[end - 1].isWhitespace()) end--
                seg.setLength(end)
            }
            flush()
            out.add(hit.id)
            i += hit.text.length
        }
        flush()
    }

    private fun bpe(normalized: String, out: MutableList<Int>) {
        val pieces = ArrayList<Int>(normalized.length + 8)

        var i = 0
        while (i < normalized.length) {
            val cp = normalized.codePointAt(i)
            i += Character.charCount(cp)
            val id = lookupChar(cp)
            if (id >= 0) {
                pieces.add(id)
            } else {
                // byte fallback: vocab에 없는 문자는 UTF-8 바이트를 <0xNN> 토큰으로 떨어뜨린다.
                for (b in String(Character.toChars(cp)).toByteArray(Charsets.UTF_8)) {
                    val id = byteIds[b.toInt() and 0xFF]
                    // v2 블롭은 <0x09>가 없다(-1). 탭은 단일 문자 토큰이라 여기까지 올 수 없다.
                    check(id >= 0) { "바이트 폴백 토큰이 없다: 0x%02X".format(b.toInt() and 0xFF) }
                    pieces.add(id)
                }
            }
        }

        while (pieces.size > 1) {
            var bestRank = Int.MAX_VALUE
            var bestAt = -1
            var bestMerged = -1
            for (p in 0 until pieces.size - 1) {
                val v = lookupMerge(pieces[p], pieces[p + 1])
                if (v != NO_MERGE) {
                    val rank = (v ushr 32).toInt()
                    if (rank < bestRank) {
                        bestRank = rank
                        bestAt = p
                        bestMerged = (v and 0xFFFFFFFFL).toInt()
                    }
                }
            }
            if (bestAt < 0) break
            pieces[bestAt] = bestMerged
            pieces.removeAt(bestAt + 1)
        }
        out.addAll(pieces)
    }

    /** 코드포인트 → 토큰 ID. 블롭이 코드포인트 오름차순이라 이진 탐색으로 찾는다. */
    private fun lookupChar(codePoint: Int): Int {
        var lo = 0
        var hi = charCodepoints.size - 1
        while (lo <= hi) {
            val mid = (lo + hi) ushr 1
            val v = charCodepoints[mid]
            when {
                v < codePoint -> lo = mid + 1
                v > codePoint -> hi = mid - 1
                else -> return charIds[mid]
            }
        }
        return -1
    }

    /**
     * (idA, idB) → (rank shl 32) or mergedId. 없으면 [NO_MERGE].
     *
     * 51만 개를 `HashMap<Long, Long>`으로 들면 박싱 때문에 수십 MB가 된다. 선형 탐사
     * 개방주소법에 원시 배열을 쓰면 같은 데이터가 16MB 남짓이다.
     */
    private fun lookupMerge(a: Int, b: Int): Long {
        val key = mergeKey(a, b)
        var idx = hash(key) and (mergeKeys.size - 1)
        while (true) {
            val k = mergeKeys[idx]
            if (k == 0L) return NO_MERGE
            if (k == key) return mergeValues[idx]
            idx = (idx + 1) and (mergeKeys.size - 1)
        }
    }

    companion object {
        private const val NO_MERGE = -1L

        // 키 0은 '빈 칸'으로 쓰므로 ID에 1을 더해 저장한다. 안 그러면 (0,0) 쌍과 구분이 안 된다.
        private fun mergeKey(a: Int, b: Int): Long =
            ((a + 1).toLong() shl 32) or (b + 1).toLong()

        private fun hash(key: Long): Int {
            var h = key * -0x61c8864680b583ebL
            h = h xor (h ushr 32)
            return (h.toInt() and 0x7fffffff)
        }

        fun load(blob: File): GemmaBpeTokenizer {
            val buf = ByteBuffer.wrap(blob.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
            val magic = ByteArray(4).also { buf.get(it) }
            check(String(magic) == "GBPE") { "블롭 매직이 다르다: ${String(magic)}" }
            val version = buf.int
            check(version == 1 || version == 2) { "모르는 블롭 버전 $version — build_tokenizer_blob.py와 맞출 것" }

            val bos = buf.int
            val eos = buf.int

            val charCount = buf.int
            val codepoints = IntArray(charCount)
            val ids = IntArray(charCount)
            for (i in 0 until charCount) {
                codepoints[i] = buf.int
                ids[i] = buf.int
            }

            val byteIds = IntArray(256) { buf.int }

            val mergeCount = buf.int
            var cap = 1
            while (cap < mergeCount * 2) cap = cap shl 1
            val keys = LongArray(cap)
            val values = LongArray(cap)
            for (rank in 0 until mergeCount) {
                val a = buf.int
                val b = buf.int
                val merged = buf.int
                val key = mergeKey(a, b)
                var idx = hash(key) and (cap - 1)
                while (keys[idx] != 0L) idx = (idx + 1) and (cap - 1)
                keys[idx] = key
                values[idx] = (rank.toLong() shl 32) or (merged.toLong() and 0xFFFFFFFFL)
            }

            if (version == 1) return GemmaBpeTokenizer(bos, eos, ids, codepoints, byteIds, keys, values)

            // v2 꼬리: flags(bit0 = metaspace) + added token 표. (이 데모의 gemma_tokenizer.bin은 v1이라 쓰지 않음.)
            val flags = buf.int
            val added = List(buf.int) {
                val id = buf.int
                val lstrip = buf.int != 0
                val cps = IntArray(buf.int) { buf.int }
                AddedToken(String(cps, 0, cps.size), id, lstrip)
            }
            return GemmaBpeTokenizer(bos, eos, ids, codepoints, byteIds, keys, values, flags and 1 != 0, added)
        }
    }
}
