package io.github.kakasolg.phototextsearch

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.BitmapFactory
import android.net.ConnectivityManager
import android.net.TrafficStats
import android.os.BatteryManager
import android.os.Debug
import android.os.PowerManager
import android.os.Process
import android.provider.Settings
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import java.io.File
import kotlin.coroutines.resume
import kotlinx.coroutines.suspendCancellableCoroutine
import org.json.JSONArray
import org.json.JSONObject

/**
 * Photo/screenshot text demo, step 3 (README): everything on the phone.
 *
 * photo → ML Kit OCR → rows ([DemoOcrBench.rows]) → EmbeddingGemma seq64 chunks (CompiledModel CPU) → per-photo max score
 * question → same embedder (query prefix) → top 3 photos → Gemma 4 E2B: yes/no gate, then a cited answer or "없음".
 *
 * The index (OCR rows + chunk vectors) is cached in the app's files dir so the screen opens ready; [index] with
 * `force = true` rebuilds it and is what the timing run measures.
 */
@OptIn(ExperimentalApi::class)
class PhotoTextSearchDemo(private val appContext: Context) : AutoCloseable {

    class Photo(val id: String, val path: String, val rows: List<String>, val chunks: List<FloatArray>)

    class Hit(val photo: Photo, val score: Float)

    class SearchResult(val hits: List<Hit>, val embedMs: Double, val scoreMs: Double, val tokenizeMs: Double = 0.0)

    class Answer(
        val gateRaw: String,
        val gateYes: Boolean,
        val raw: String,
        /** Photo ids the model cited; empty = abstained ("없음"); null = unparseable. */
        val cited: Set<String>?,
        val text: String,
        val gateMs: Double,
        val answerMs: Double,
        val firstTokenMs: Double?,
        /** v4 store check: raw verdict per cited photo id (empty before v4), its total ms, and the citations before it. */
        val verifyRaw: Map<String, String> = emptyMap(),
        val verifyMs: Double = 0.0,
        val citedBeforeVerify: Set<String>? = cited,
    )

    var photos: List<Photo> = emptyList()
        private set

    private var tokenizer: GemmaBpeTokenizer? = null
    private var embedder: EmbeddingGemmaEmbedder? = null
    private var engine: Engine? = null

    var embedderLoadMs = 0.0
        private set
    var llmLoadMs = 0.0
        private set

    private val indexFile = File(appContext.filesDir, "photo_demo_index.json")

    fun loadEmbedder() {
        if (embedder != null) return
        val t0 = System.nanoTime()
        val tok = GemmaBpeTokenizer.load(File(TOKENIZER_PATH))
        tokenizer = tok
        embedder = EmbeddingGemmaEmbedder(File(EMBEDDER_PATH), tok).also { it.initialize() }
        embedderLoadMs = (System.nanoTime() - t0) / 1e6
    }

    fun loadLlm() {
        if (engine != null) return
        val t0 = System.nanoTime()
        // constrained/speculative decoding off, same as RagAnswerProbeViewModel (GPU driver crash history).
        ExperimentalFlags.enableSpeculativeDecoding = false
        engine = Engine(
            EngineConfig(
                modelPath = LLM_PATH,
                backend = Backend.GPU(),
                maxNumTokens = MAX_TOKENS,
                cacheDir = appContext.getExternalFilesDir(null)?.absolutePath,
            ),
        ).also { it.initialize() }
        llmLoadMs = (System.nanoTime() - t0) / 1e6
    }

    /** Loads the cached index when it covers exactly [paths]; returns false when it has to be built. */
    fun loadCachedIndex(paths: List<String>): Boolean {
        if (!indexFile.exists()) return false
        val o = JSONObject(indexFile.readText())
        if (o.optString("embedder") != EMBEDDER_PATH) return false
        val arr = o.getJSONArray("photos")
        val loaded = (0 until arr.length()).map { i ->
            val p = arr.getJSONObject(i)
            val rows = p.getJSONArray("rows").let { r -> (0 until r.length()).map(r::getString) }
            val chunks = p.getJSONArray("chunks").let { c ->
                (0 until c.length()).map { j -> c.getJSONArray(j).let { v -> FloatArray(v.length()) { k -> v.getDouble(k).toFloat() } } }
            }
            Photo(p.getString("id"), p.getString("path"), rows, chunks)
        }
        if (loaded.map { it.path } != paths) return false
        photos = loaded
        return true
    }

    /**
     * OCR + embed every photo. Returns per-photo timings (ms) as JSON for the run record.
     * Recognizer: the bundled Korean ML Kit model, original resolution, one pass (same as Field Lab / DemoOcrBench).
     */
    fun index(paths: List<String>, onProgress: (Int, Int) -> Unit): JSONArray {
        val emb = checkNotNull(embedder) { "loadEmbedder() first" }
        val recognizer = TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build())
        val timings = JSONArray()
        val built = ArrayList<Photo>(paths.size)
        try {
            paths.forEachIndexed { i, path ->
                val bmp = BitmapFactory.decodeFile(path) ?: return@forEachIndexed
                val t0 = System.nanoTime()
                val result = try {
                    Tasks.await(recognizer.process(InputImage.fromBitmap(bmp, 0)))
                } finally {
                    bmp.recycle()
                }
                val ocrMs = (System.nanoTime() - t0) / 1e6
                val rows = DemoOcrBench.rows(result)
                val t1 = System.nanoTime()
                val doc = emb.embedDocument(rows.joinToString(" ").ifBlank { " " })
                val embedMs = (System.nanoTime() - t1) / 1e6
                built += Photo(File(path).nameWithoutExtension, path, rows, doc.chunkVectors)
                timings.put(JSONObject().put("id", File(path).nameWithoutExtension).put("ocr_ms", ocrMs)
                    .put("embed_ms", embedMs).put("chunks", doc.chunkVectors.size))
                onProgress(i + 1, paths.size)
            }
        } finally {
            recognizer.close()
        }
        photos = built
        indexFile.writeText(JSONObject().put("embedder", EMBEDDER_PATH).put("photos", JSONArray().apply {
            built.forEach { p ->
                put(JSONObject().put("id", p.id).put("path", p.path).put("rows", JSONArray(p.rows))
                    .put("chunks", JSONArray().apply { p.chunks.forEach { v -> put(JSONArray(v.map { it.toDouble() })) } }))
            }
        }).toString())
        return timings
    }

    fun search(question: String, k: Int = TOP_PHOTOS): SearchResult {
        val emb = checkNotNull(embedder) { "loadEmbedder() first" }
        val t0 = System.nanoTime()
        val ids = checkNotNull(tokenizer).encode(EmbeddingGemmaEmbedder.TaskPrefix.QUERY + question)
        val tt = System.nanoTime()
        val q = emb.embedTokens(ids)
        val t1 = System.nanoTime()
        val hits = photos.map { Hit(it, DocumentChunker.maxScore(q, it.chunks)) }.sortedByDescending { it.score }.take(k)
        val t2 = System.nanoTime()
        return SearchResult(hits, (t1 - t0) / 1e6, (t2 - t1) / 1e6, (tt - t0) / 1e6)
    }

    /** Embedder speed with nothing else in the call: the same 64-token input [n] times (ms each), and the backend in use. */
    fun embedMicrobench(n: Int = 20): JSONObject {
        val emb = checkNotNull(embedder) { "loadEmbedder() first" }
        val ids = IntArray(emb.sequenceLength) { i -> if (i == 0) 2 else 1000 + i }
        val ms = JSONArray()
        repeat(n) {
            val t = System.nanoTime(); emb.embedTokens(ids); ms.put((System.nanoTime() - t) / 1e6)
        }
        return JSONObject().put("backend", emb.activeBackend.name).put("ms", ms)
    }

    /**
     * ── 기권은 점수가 아니라 게이트로 ──────────────────────────────
     *  · 2단계 PC 기준선에서 "답 없음" 질문의 1위 점수(0.45~0.60)가 맞힌 질문(최소 0.44, 중앙 0.60)과 겹쳐서
     *    임계값으로는 기권을 못 한다. 앞선 검색·답 실험에서 전 항목 만점이었던 방식을 그대로 쓴다:
     *    먼저 예/아니오만 묻고, '아니오'면 답을 만들지 않는다.
     *  · 사진 번호는 [사진1]~[사진3]으로 바꿔 넣는다. 파일 id(m01, r03…)를 그대로 주면 종류가 id에 드러나서
     *    모델이 글자 대신 id로 고를 수 있다.
     *  · 데모 데이터에는 이름이 비슷한 다른 가게가 일부러 있어서(한빛식당/한빛분식 등) "가게 이름까지 맞는지"를
     *    규칙에 넣었다. 이 프롬프트는 60문항을 돌리기 전에 고정했다.
     *
     * ── v2 (2026-10-15): 게이트가 부정형 답을 "근거 없음"으로 읽는 문제 ──────────────
     *  · v1 기권 5~6건이 전부 같은 모양이었다: "Can I bring my own wine?" + "Corkage fee $20", "환불돼?" + "not refundable",
     *    "Can I take my dog?" + "실내 동반 불가", "overnight parking rule" + "No overnight parking", "Thanksgiving?" + "CLOSED".
     *    정답 사진이 1위인데 게이트가 '아니오'. 게이트에 "답할 사진이 있나"를 물었는데 모델은 **사용자의 예/아니오 질문 자체**에
     *    답한 것으로 보인다 — 앞선 메모 검색 실험의 극성 함정이 다른 모양으로 나온 것.
     *  · 고친 것: ① 게이트 출력 단어를 '예/아니오'에서 '있음/없음'으로 바꿔 사용자 질문의 예/아니오와 겹치지 않게 했다.
     *    ② "너는 질문에 답하는 게 아니라 정보가 있는지만 본다", "답이 안 된다·No·not·closed 같은 부정이어도 정보가 있으면 있음"을
     *    명시했다. ③ 답 단계에도 같은 함정이 있을 수 있어 "부정인 답도 그 사진을 근거로 적어라" 한 줄을 넣었다.
     *  · 나머지(가게 이름 확인, 다른 사진 인용)는 그대로 둔다 — 한 번에 하나만 바꿔야 무엇이 효과였는지 보인다.
     *    v1은 [PromptVersion.V1]로 그대로 남긴다(실패 갤러리 재현용).
     *
     * ── v3 (2026-10-15): 가게 이름 규칙이 종류·다른 언어로 부른 질문을 막는 문제 ──────────
     *  · v2에서 남은 기권 4건은 극성이 아니라 질문이 가게를 이름 없이("the pho place", "태국 식당", "아침 식당") 또는 다른 언어로
     *    ("Daon Tea Room" ↔ "다온 찻집") 부른 경우였다. "이름이 질문과 다르면 치지 않는다"가 이름이 없는 질문까지 막은 것으로 본다.
     *  · 고친 것: 이름 규칙을 "질문이 가게를 어떻게 부르는지"로 나눴다 — 이름으로 부르면 같은 가게만(영어·한글 표기는 같은 이름,
     *    비슷하지만 다른 가게는 다른 가게), 종류·특징으로 부르면 그 종류에 맞는 가게. 게이트와 답 단계의 이름 줄만 바꾸고 v2의 나머지는
     *    그대로 둔다.
     *  · 예시 이름(Haneul/하늘, 미래제과/미래분식, 아이스크림 가게, the burger place)은 60문항·80장 어디에도 없는 말로 골랐다 —
     *    시험 문항의 말을 프롬프트에 넣으면 그 문항만 맞히게 된다.
     *
     * ── v4 (2026-10-15): 답한 뒤 "근거 사진이 질문의 가게 것인가"를 따로 확인 ──────────────
     *  · A05("Moonrise Donuts happy hour", 데이터에 없음)는 v1~v3 모두 게이트를 넘고 Corner Slice 공지를 인용했다. 답 문장은
     *    "Corner Slice의 해피아워…"일 때도, **"Moonrise Donuts의 행복 시간…"처럼 질문의 가게 이름을 붙일 때도** 있었다. 그래서 답에
     *    적힌 이름만 보면 놓친다 — 확인 대상은 **인용한 사진의 가게**다.
     *  · v3 프롬프트는 그대로 두고, 답이 사진을 인용했을 때만 사진마다 한 번 더 묻는다: 질문 + 그 사진 한 장의 글자 + 답 →
     *    `같음/다름` 한 단어. 출력 단어를 게이트(있음/없음)·사용자 질문(예/아니오)과 겹치지 않게 골랐다. '다름'인 사진은 인용에서
     *    빼고, 남는 사진이 없으면 기권으로 바꾼다. 이름 판단 기준은 [NAME_RULE_V3]를 그대로 쓴다.
     *  · 대가: 답한 질문마다 Gemma 호출이 한 번 늘어난다. 맞는 답을 '다름'으로 잘못 버리는 경우도 같이 센다.
     *  · 결과: 108번 모두 '같음'(A05 포함) — 효과 없음. 그래서 v5는 이 단계를 빼고 코드로 대조한다.
     *
     * ── v5 (2026-10-15): 가게 이름을 코드로 대조 ([PhotoStoreNames]) ──────────────
     *  · v3 프롬프트 + 답 뒤에 모델 없이 확인한다. 질문이 아는 가게를 부르는데, 인용한 사진 글자에 그 가게 이름이 없고 사진은 다른
     *    아는 가게의 것이면 그 인용을 뺀다. 남는 사진이 없으면 기권.
     *  · 실행 전에 지난 8회차 출력에 같은 규칙을 돌려 봤다(demo/photo_text_search/store_check_dryrun.py): A05만 8번 모두 걸리고 맞는 답은
     *    0건 버림. 첫 규칙은 앱 화면 제목 "예약 완료"를 가게 이름으로 잡아 C02를 잘못 버려서, "Details" 아래 첫 줄(화면 제목)은 건너뛰게 했다.
     */
    enum class PromptVersion { V1, V2, V3, V4, V5 }

    private var storeNames: PhotoStoreNames? = null

    suspend fun answer(question: String, hits: List<Hit>, version: PromptVersion = PromptVersion.V1): Answer {
        val labels = hits.mapIndexed { i, h -> "사진${i + 1}" to h.photo.id }
        val records = buildString {
            hits.forEachIndexed { i, h ->
                appendLine("[사진${i + 1}]")
                h.photo.rows.forEach { appendLine(it) }
                appendLine()
            }
        }
        val gatePrompt = buildString {
            appendLine("아래는 내 폰에 있는 사진에서 읽은 글자다. 글자 인식이라 일부 틀린 글자가 있을 수 있다.")
            appendLine()
            append(records)
            appendLine("질문: $question")
            appendLine()
            if (version == PromptVersion.V1) {
                appendLine("위 사진들을 하나씩 살펴봐라. 표현이나 언어가 달라도 뜻이 같으면 그 사진은 질문에 답하는 것으로 친다.")
                appendLine("가게나 장소 이름이 질문과 다르면(이름이 비슷한 다른 가게 포함) 답으로 치지 않는다.")
                appendLine("질문에 답할 수 있는 사진이 하나라도 있으면 '예', 하나도 없으면 '아니오'다.")
                appendLine("다른 말은 하지 말고 '예' 또는 '아니오' 한 단어만 출력해라.")
            } else {
                appendLine("위 사진들을 하나씩 살펴봐라. 너는 질문에 답하는 것이 아니다. 질문에 답하는 데 필요한 정보가")
                appendLine("사진 글자에 있는지만 판단한다.")
                appendLine("질문이 '~돼?', 'Can I ...?', 'Is ...?'처럼 예/아니오로 묻더라도, 사진의 답이 '안 된다', '불가', 'No ...',")
                appendLine("'not ...', 'closed'처럼 부정이면 그것도 답이다 — 그 정보가 사진에 있으면 '있음'이다.")
                appendLine("표현이나 언어가 달라도 뜻이 같으면 정보가 있는 것으로 친다.")
                if (version == PromptVersion.V2) {
                    appendLine("가게나 장소 이름이 질문과 다르면(이름이 비슷한 다른 가게 포함) 그 사진은 치지 않는다.")
                } else {
                    NAME_RULE_V3.forEach(::appendLine)
                }
                appendLine("정보가 있는 사진이 하나라도 있으면 '있음', 하나도 없으면 '없음'이다.")
                appendLine("다른 말은 하지 말고 '있음' 또는 '없음' 한 단어만 출력해라.")
            }
        }
        val (gateRaw, gateMs, _) = ask(gatePrompt)
        val gateYes = if (version == PromptVersion.V1) !GATE_NO.containsMatchIn(gateRaw) else !gateRaw.contains("없")
        if (!gateYes) {
            return Answer(gateRaw, false, "", emptySet(), "", gateMs, 0.0, null)
        }
        val answerPrompt = buildString {
            appendLine("아래는 내 폰에 있는 사진에서 읽은 글자다. 글자 인식이라 일부 틀린 글자가 있을 수 있다.")
            appendLine()
            append(records)
            appendLine("질문: $question")
            appendLine()
            appendLine("위 사진의 글자만 근거로 답해라. 사진에 없는 내용은 지어내지 마라.")
            if (version.ordinal >= PromptVersion.V3.ordinal) {
                NAME_RULE_V3.forEach(::appendLine)
                appendLine("이 기준에 맞는 사진만 근거로 써라.")
            } else {
                appendLine("가게나 장소 이름이 질문과 맞는 사진만 근거로 써라. 이름이 비슷한 다른 가게의 사진은 근거가 아니다.")
            }
            appendLine("맞는 사진이 하나도 없으면 근거를 '없음'으로 적어라.")
            if (version != PromptVersion.V1) {
                appendLine("답이 '안 된다', '불가', 'No', 'closed'처럼 부정이어도 사진에 그 내용이 있으면 근거에 그 사진을 적어라.")
            }
            appendLine("가격·시간·번호는 사진에 적힌 그대로 옮겨라.")
            appendLine()
            appendLine("아래 두 줄 형식으로만 답해라.")
            appendLine("근거: ${labels.first().first}")
            appendLine("답변: <질문과 같은 언어로 한 문장>")
        }
        val (raw, answerMs, firstMs) = ask(answerPrompt)
        val citeLine = CITATION_LINE.find(raw)?.groupValues?.get(1)
        val cited: Set<String>? = when {
            citeLine == null -> null
            citeLine.contains("없음") || citeLine.contains("없다") -> emptySet()
            else -> labels.filter { (label, _) -> Regex("""$label(?!\d)""").containsMatchIn(citeLine) }.map { it.second }.toSet()
                .ifEmpty { null }
        }
        val text = ANSWER_LINE.find(raw)?.groupValues?.get(1)?.trim() ?: raw
        if (version == PromptVersion.V5 && !cited.isNullOrEmpty()) {
            val t0 = System.nanoTime()
            val names = storeNames?.takeIf { it.photos === photos } ?: PhotoStoreNames(photos).also { storeNames = it }
            val reasons = LinkedHashMap<String, String>()
            val kept = cited.filter { id ->
                val (ok, why) = names.keep(question, id)
                reasons[id] = (if (ok) "keep: " else "drop: ") + why
                ok
            }.toSet()
            return Answer(gateRaw, true, raw, kept, text, gateMs, answerMs, firstMs, reasons, (System.nanoTime() - t0) / 1e6, cited)
        }
        if (version != PromptVersion.V4 || cited.isNullOrEmpty()) {
            return Answer(gateRaw, true, raw, cited, text, gateMs, answerMs, firstMs)
        }
        val verdicts = LinkedHashMap<String, String>()
        var verifyMs = 0.0
        val kept = cited.filter { id ->
            val photo = hits.first { it.photo.id == id }.photo
            val (verdict, ms, _) = ask(storeCheckPrompt(question, photo, text))
            verifyMs += ms
            verdicts[id] = verdict
            !verdict.contains("다름")
        }.toSet()
        return Answer(gateRaw, true, raw, kept, text, gateMs, answerMs, firstMs, verdicts, verifyMs, cited)
    }

    private fun storeCheckPrompt(question: String, photo: Photo, answer: String): String = buildString {
        appendLine("아래는 내 폰에 있는 사진 한 장에서 읽은 글자다. 글자 인식이라 일부 틀린 글자가 있을 수 있다.")
        appendLine()
        appendLine("[사진]")
        photo.rows.forEach { appendLine(it) }
        appendLine()
        appendLine("질문: $question")
        appendLine("답: $answer")
        appendLine()
        appendLine("이 사진이 질문이 묻는 가게·장소의 것인지만 판단해라. 답이 맞는지는 보지 않는다.")
        appendLine("먼저 질문이 어느 가게·장소를 묻는지 보고, 다음에 이 사진이 어느 가게·장소의 것인지 사진 글자에서 찾아라.")
        NAME_RULE_V3.forEach(::appendLine)
        appendLine("답 문장에 질문의 가게 이름이 적혀 있어도, 사진 글자가 다른 가게의 것이면 다른 가게다.")
        appendLine("같은 가게·장소면 '같음', 다른 가게·장소면 '다름'이다.")
        appendLine("다른 말은 하지 말고 '같음' 또는 '다름' 한 단어만 출력해라.")
    }

    /** One fresh Conversation per prompt (litertlm-android second-turn bug,). */
    private suspend fun ask(prompt: String): Triple<String, Double, Double?> {
        val eng = checkNotNull(engine) { "loadLlm() first" }
        val conv = eng.createConversation(
            ConversationConfig(samplerConfig = SamplerConfig(topK = 1, topP = 1.0, temperature = 0.0)),
        )
        val t0 = System.nanoTime()
        var first: Double? = null
        return try {
            val text = suspendCancellableCoroutine { cont ->
                val buf = StringBuilder()
                try {
                    conv.sendMessageAsync(
                        Contents.of(listOf(Content.Text(prompt))),
                        object : MessageCallback {
                            override fun onMessage(message: Message) {
                                if (first == null) first = (System.nanoTime() - t0) / 1e6
                                buf.append(message.toString())
                            }

                            override fun onDone() {
                                if (cont.isActive) cont.resume(buf.toString().trim())
                            }

                            override fun onError(throwable: Throwable) {
                                if (cont.isActive) cont.resume("(오류) ${throwable::class.simpleName}: ${throwable.message}")
                            }
                        },
                        emptyMap(),
                    )
                } catch (e: Throwable) {
                    if (cont.isActive) cont.resume("(동기 예외) ${e::class.simpleName}: ${e.message}")
                }
            }
            Triple(text, (System.nanoTime() - t0) / 1e6, first)
        } finally {
            conv.close()
        }
    }

    override fun close() {
        engine?.close(); engine = null
        embedder?.close(); embedder = null
    }

    /** What the run record needs to show "no network": airplane mode, active network, and this app's own bytes. */
    object Offline {
        fun snapshot(context: Context): JSONObject {
            val uid = Process.myUid()
            val cm = context.getSystemService(ConnectivityManager::class.java)
            return JSONObject()
                .put("airplane_mode", Settings.Global.getInt(context.contentResolver, Settings.Global.AIRPLANE_MODE_ON, 0) == 1)
                .put("active_network", cm?.activeNetwork?.toString() ?: JSONObject.NULL)
                .put("uid_tx_bytes", TrafficStats.getUidTxBytes(uid))
                .put("uid_rx_bytes", TrafficStats.getUidRxBytes(uid))
        }
    }

    /**
     * ── 배터리 값은 케이블을 뺐을 때만 의미가 있다 ──────────────────────
     *  · USB가 꽂혀 있으면 충전기가 부하를 대신 대서 배터리 전류·잔량 변화가 소비를 보여 주지 않는다. 샘플마다
     *    plugged를 같이 적고, 채점 쪽은 plugged = 0인 구간만 쓴다.
     *  · Pixel의 CURRENT_NOW는 µA, 방전이 음수다. 에너지는 |I|·V를 1초 간격으로 더한 근사다.
     */
    object Battery {
        fun sample(context: Context, t0: Long): JSONObject {
            val bm = context.getSystemService(BatteryManager::class.java)
            val sticky = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            val pm = context.getSystemService(PowerManager::class.java)
            return JSONObject()
                .put("t_ms", (System.nanoTime() - t0) / 1_000_000)
                .put("current_ua", bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW))
                .put("charge_uah", bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER))
                .put("voltage_mv", sticky?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, -1) ?: -1)
                .put("plugged", sticky?.getIntExtra(BatteryManager.EXTRA_PLUGGED, -1) ?: -1)
                .put("level", bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY))
                .put("thermal", pm.currentThermalStatus)
                .put("pss_kb", Debug.getPss())
        }
    }

    companion object {
        const val EMBEDDER_PATH = "/data/local/tmp/embeddinggemma-300m_seq64_q8.tflite"
        const val TOKENIZER_PATH = "/data/local/tmp/gemma_tokenizer.bin"
        const val LLM_PATH = "/data/local/tmp/gemma-4-E2B-it.litertlm"
        const val IMAGES_LIST = "/data/local/tmp/demo_images/list.txt"
        const val QUESTIONS_PATH = "/data/local/tmp/demo_images/questions.json"
        const val TOP_PHOTOS = 3
        private const val MAX_TOKENS = 4096
        private val NAME_RULE_V3 = listOf(
            "가게나 장소는 질문이 어떻게 부르는지에 따라 판단한다.",
            "- 질문에 가게 이름이 있으면 그 이름의 가게 사진만 친다. 영어 표기와 한글 표기가 같은 이름이면 같은 가게다",
            "  (예: 'Haneul' = '하늘'). 이름이 비슷해도 다른 가게는 다른 가게다(예: '미래제과'와 '미래분식').",
            "- 질문이 이름 대신 종류나 특징으로 부르면(예: '아이스크림 가게', 'the burger place') 그 종류·특징에 맞는 가게의 사진을 친다.",
        )
        private val GATE_NO = Regex("""아니(오|요)|없|\bno\b""", RegexOption.IGNORE_CASE)
        private val CITATION_LINE = Regex("""근거\s*[:：]\s*(.+)""")
        private val ANSWER_LINE = Regex("""답변\s*[:：]\s*(.+)""", RegexOption.DOT_MATCHES_ALL)
    }
}
