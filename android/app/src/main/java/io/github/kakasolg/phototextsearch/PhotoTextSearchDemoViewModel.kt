package io.github.kakasolg.phototextsearch

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

class PhotoTextSearchDemoViewModel(private val appContext: Context) : ViewModel() {

    data class QueryState(
        val question: String = "",
        val hits: List<PhotoTextSearchDemo.Hit> = emptyList(),
        val answer: PhotoTextSearchDemo.Answer? = null,
        val searchMs: Double = 0.0,
        val running: Boolean = false,
    )

    private val demo = PhotoTextSearchDemo(appContext)

    private val _status = MutableStateFlow("준비 전")
    val status: StateFlow<String> = _status.asStateFlow()

    private val _busy = MutableStateFlow(false)
    val busy: StateFlow<Boolean> = _busy.asStateFlow()

    private val _query = MutableStateFlow(QueryState())
    val query: StateFlow<QueryState> = _query.asStateFlow()

    private val _offline = MutableStateFlow("")
    val offline: StateFlow<String> = _offline.asStateFlow()

    private val _promptVersion = MutableStateFlow(PhotoTextSearchDemo.PromptVersion.V5)
    val promptVersion: StateFlow<PhotoTextSearchDemo.PromptVersion> = _promptVersion.asStateFlow()

    fun setPromptVersion(v: PhotoTextSearchDemo.PromptVersion) {
        _promptVersion.value = v
    }

    private val baseline = PhotoTextSearchDemo.Offline.snapshot(appContext)

    init {
        viewModelScope.launch {
            while (isActive) {
                val now = PhotoTextSearchDemo.Offline.snapshot(appContext)
                _offline.value = "비행기 모드 ${if (now.getBoolean("airplane_mode")) "켜짐" else "꺼짐"} · " +
                    "이 화면 이후 앱 송신 ${now.getLong("uid_tx_bytes") - baseline.getLong("uid_tx_bytes")} B · " +
                    "수신 ${now.getLong("uid_rx_bytes") - baseline.getLong("uid_rx_bytes")} B"
                delay(1000)
            }
        }
    }

    private fun imagePaths(): List<String> =
        File(PhotoTextSearchDemo.IMAGES_LIST).readLines().map { it.trim() }.filter { it.isNotEmpty() }

    fun prepare(forceReindex: Boolean = false) {
        if (_busy.value) return
        _busy.value = true
        viewModelScope.launch {
            try {
                withContext(Dispatchers.Default) {
                    _status.value = "임베더 로드 중"
                    demo.loadEmbedder()
                    val paths = imagePaths()
                    if (forceReindex || !demo.loadCachedIndex(paths)) {
                        demo.index(paths) { done, total -> _status.value = "색인 중 $done/$total (글자 읽기 + 임베딩)" }
                    }
                    _status.value = "Gemma 로드 중"
                    demo.loadLlm()
                }
                _status.value = "준비됨 · 사진 ${demo.photos.size}장"
            } catch (e: Exception) {
                _status.value = "실패: ${e::class.simpleName}: ${e.message}"
            } finally {
                _busy.value = false
            }
        }
    }

    fun ask(question: String) {
        if (_busy.value || question.isBlank()) return
        _busy.value = true
        _query.value = QueryState(question = question, running = true)
        viewModelScope.launch {
            try {
                val res = withContext(Dispatchers.Default) { demo.search(question) }
                _query.value = _query.value.copy(hits = res.hits, searchMs = res.embedMs + res.scoreMs)
                val ans = demo.answer(question, res.hits, _promptVersion.value)
                _query.value = _query.value.copy(answer = ans, running = false)
            } catch (e: Exception) {
                _status.value = "실패: ${e::class.simpleName}: ${e.message}"
                _query.value = _query.value.copy(running = false)
            } finally {
                _busy.value = false
            }
        }
    }

    /**
     * The 60 fixed questions end to end, written to `files/photo_demo/run_<tag>.json`.
     * [delaySec] gives time to pull the USB cable before anything is measured (battery numbers are only valid unplugged).
     * Models are loaded inside the run so their load time and memory are part of the record.
     */
    fun runBatch(tag: String, reindex: Boolean, delaySec: Int) {
        if (_busy.value) return
        _busy.value = true
        viewModelScope.launch {
            try {
                for (s in delaySec downTo 1) {
                    _status.value = "$s 초 뒤 시작 (케이블을 빼면 배터리도 잰다)"
                    delay(1000)
                }
                val t0 = System.nanoTime()
                val samples = JSONArray()
                val phases = JSONArray()
                fun mark(name: String) = phases.put(JSONObject().put("phase", name).put("t_ms", (System.nanoTime() - t0) / 1_000_000))
                val sampler = launch(Dispatchers.Default) {
                    while (isActive) {
                        samples.put(PhotoTextSearchDemo.Battery.sample(appContext, t0))
                        delay(1000)
                    }
                }
                val version = _promptVersion.value
                val out = JSONObject().put("tag", tag).put("prompt_version", version.name).put("device", android.os.Build.MODEL)
                    .put("offline_start", PhotoTextSearchDemo.Offline.snapshot(appContext))
                val items = JSONArray()
                withContext(Dispatchers.Default) {
                    demo.close()
                    mark("load_embedder")
                    _status.value = "임베더 로드"
                    demo.loadEmbedder()
                    out.put("embedder_load_ms", demo.embedderLoadMs)
                    out.put("embed_micro_before", demo.embedMicrobench())
                    val paths = imagePaths()
                    mark("index")
                    if (reindex || !demo.loadCachedIndex(paths)) {
                        out.put("index", demo.index(paths) { d, n -> _status.value = "색인 $d/$n" })
                    }
                    out.put("index_ms", (System.nanoTime() - t0) / 1e6 - demo.embedderLoadMs)
                    mark("load_llm")
                    _status.value = "Gemma 로드"
                    demo.loadLlm()
                    out.put("llm_load_ms", demo.llmLoadMs).put("pss_kb_both_loaded", android.os.Debug.getPss())
                    out.put("embed_micro_after_llm", demo.embedMicrobench())
                    mark("questions")
                    val qs = JSONArray(File(PhotoTextSearchDemo.QUESTIONS_PATH).readText())
                    for (i in 0 until qs.length()) {
                        val q = qs.getJSONObject(i)
                        _status.value = "질문 ${i + 1}/${qs.length()}"
                        val res = demo.search(q.getString("q"))
                        val ans = demo.answer(q.getString("q"), res.hits, version)
                        items.put(JSONObject().put("id", q.getString("id")).put("q", q.getString("q"))
                            .put("top3", JSONArray(res.hits.map { it.photo.id }))
                            .put("scores", JSONArray(res.hits.map { it.score.toDouble() }))
                            .put("query_embed_ms", res.embedMs).put("query_tokenize_ms", res.tokenizeMs).put("score_ms", res.scoreMs)
                            .put("gate_raw", ans.gateRaw).put("gate_yes", ans.gateYes)
                            .put("raw", ans.raw).put("answer", ans.text)
                            .put("cited", ans.cited?.let { JSONArray(it.toList()) } ?: JSONObject.NULL)
                            .put("gate_ms", ans.gateMs).put("answer_ms", ans.answerMs)
                            .put("verify", JSONObject(ans.verifyRaw.toMap())).put("verify_ms", ans.verifyMs)
                            .put("cited_before_verify", ans.citedBeforeVerify?.let { JSONArray(it.toList()) } ?: JSONObject.NULL)
                            .put("first_token_ms", ans.firstTokenMs ?: JSONObject.NULL))
                    }
                    mark("done")
                }
                sampler.cancel()
                samples.put(PhotoTextSearchDemo.Battery.sample(appContext, t0))
                out.put("offline_end", PhotoTextSearchDemo.Offline.snapshot(appContext))
                    .put("total_ms", (System.nanoTime() - t0) / 1e6)
                    .put("phases", phases).put("battery", samples).put("items", items)
                val dir = appContext.getExternalFilesDir("photo_demo")!!.apply { mkdirs() }
                val name = if (version == PhotoTextSearchDemo.PromptVersion.V1) "run_$tag.json" else "run_${tag}_${version.name.lowercase()}.json"
                File(dir, name).writeText(out.toString(1))
                _status.value = "60문항 끝 → photo_demo/$name"
                android.util.Log.i("PhotoDemo", "batch done: $name")
            } catch (e: Exception) {
                _status.value = "실패: ${e::class.simpleName}: ${e.message}"
                android.util.Log.e("PhotoDemo", "batch failed", e)
            } finally {
                _busy.value = false
            }
        }
    }

    override fun onCleared() {
        demo.close()
    }
}
