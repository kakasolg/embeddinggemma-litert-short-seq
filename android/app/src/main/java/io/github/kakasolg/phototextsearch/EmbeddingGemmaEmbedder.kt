package io.github.kakasolg.phototextsearch

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter

/**
 * EmbeddingGemma-300M(768차원) 텍스트 임베더. 온디바이스 RAG 트랙의 1단계로, 검색·저장은
 * 아직 없고 "임베딩 한 번을 제대로 뽑을 수 있는가"만 담당한다.
 *
 * ── 왜 litertlm이 아니라 litert인가 ─────────────────────
 * 이 프로젝트가 이미 쓰는 `litertlm-android`는 `.litertlm` LLM 번들 전용 생성 런타임이라
 * 임베딩 모델을 못 돌린다. EmbeddingGemma는 `.tflite`로 배포되므로 별개 런타임이 필요하다.
 *
 * ── 1.4.x(Interpreter) vs 2.x(CompiledModel) ────────────
 * 여기서는 1.4.x의 고전 `Interpreter` API를 쓴다. litert-community 저장소에는
 * `google.tensor_g5` 같은 **벤더별 NPU 변종**이 따로 있고(Pixel 10 Pro가 여기 해당),
 * 그 가속 경로는 2.x의 CompiledModel API 쪽이다. 먼저 범용 `.tflite`를 CPU로 돌려
 * 기준선을 만들고, NPU 비교는 그 다음이다 — 사진 프로브에서 배운 순서 그대로,
 * 느린 기준선이 있어야 빠른 경로가 뭘 샀는지 말할 수 있다.
 */
/**
 * ── 토크나이저 (2026-08-11) ─────────────────────────
 * `tokenizer`를 주면 [embed]로 문자열을 바로 넣을 수 있고, 안 주면 [embedTokens]에 미리 뽑아 둔
 * 토큰 ID를 넣어야 한다. 프로브들은 후자를 쓴다 — PC 기준선과 **완전히 같은 토큰**을 먹여야
 * 점수 차이가 순수하게 tflite·양자화 탓이 되기 때문이고, 그 대조가 이 트랙의 핵심 도구다.
 *
 * 처음에는 이게 선택이 아니라 강제였다. 안드로이드 ABI로 빌드된 유일한 HF 토크나이저
 * (`io.gitlab.shubham0204:sentence-embeddings`)가 Gemma 토크나이저에서 **SIGABRT로 프로세스를
 * 죽여서** 기기 토큰화 자체가 불가능했다. [GemmaBpeTokenizer]를 직접 구현해 그 의존성을
 * 걷어냈다 — 정답지 74건을 한 토큰도 틀리지 않고 재현한다.
 */
class EmbeddingGemmaEmbedder(
    private val modelFile: File,
    private val tokenizer: GemmaBpeTokenizer? = null,
    private val numThreads: Int = 4,
    private val backend: Backend = backendOverride ?: Backend.COMPILED_CPU,
) : AutoCloseable {

    /**
     * ── 실행 경로 (2026-10-15) ─────────────────────────
     *  · [Backend.INTERPRETER]: 고전 `Interpreter`. `setNumThreads`를 줘도 Fold4(Snapdragon 8+ Gen 1)에서 실제로는 코어 하나만
     *    썼다(스레드 모니터 평균 0.92코어, 프라임 코어 87%). PC Python `Interpreter`에서는 같은 모델이 4스레드로 2배 빨라져서,
     *    모델 탓이 아니라 Android 바인딩 쪽 문제로 본다.
     *  · [Backend.COMPILED_CPU]: LiteRT 2.x `CompiledModel` + `CpuOptions(numThreads)`. 같은 벡터(최대 차이 3e-8)를 Fold4에서
     *    약 1.8배 빠르게 냈다. 시그니처 읽기·텐서 바인딩은 기존 코드를 그대로 쓰려고 `Interpreter`로 한 번 열어 읽은 뒤 닫는다.
     *  · 기본값은 [Backend.COMPILED_CPU]다(2026-10-15). Pixel 11 Pro XL·Fold4 둘 다 짧은·지속 구간에서 1.3배 이상 빠르고 벡터가
     *    비트 단위로 같아서 바꿨다(README, "Android: use CompiledModel"). `Interpreter`가 필요하면 생성자에 [Backend.INTERPRETER]를 준다.
     */
    enum class Backend { INTERPRETER, COMPILED_CPU }

    /** The backend actually in use after [initialize] (COMPILED_CPU falls back to INTERPRETER if it cannot be created). */
    var activeBackend: Backend = backend
        private set

    private var interpreter: Interpreter? = null
    private var compiled: CompiledModel? = null
    private var compiledInputs: List<TensorBuffer> = emptyList()
    private var compiledOutputs: List<TensorBuffer> = emptyList()
    private var inputIs64Bit: BooleanArray = BooleanArray(0)
    private var outputShape: IntArray = IntArray(0)
    private var outputElements: Int = 0

    /** 모델이 실제로 요구하는 입력 이름 → 인덱스. 이름 표기는 체크포인트마다 달라서 매핑해 둔다. */
    private var inputIndexByRole: Map<InputRole, Int> = emptyMap()
    /** Fixed input length of the loaded model (64, 128, 256 ...). 0 before [initialize]. */
    var sequenceLength: Int = 0
        private set
    private var embeddingOutputIndex: Int = 0

    /** 마지막 초기화에서 읽어낸 실제 시그니처. 프로브 화면이 이걸 그대로 출력한다. */
    var signature: String = "(초기화 전)"
        private set

    /**
     * 임베딩 출력 텐서의 raw 그래프 이름(`getOutputTensor(i).name()`). 참고용으로만 남긴다 —
     * `CompiledModel.getOutputBufferRequirements(outputName)`에 이 값을 그대로 넘기면
     * "Failed to find output"로 실패한다. 2.x `CompiledModel`은 시그니처 출력 키로 찾지,
     * raw 텐서 이름으로 찾지 않는다. 그 키는 [outputSignatureName]에 있다.
     */
    var outputTensorName: String = ""
        private set

    /**
     * `CompiledModel.getOutputBufferRequirements(signatureKey, outputName)`가 실제로 요구하는
     * 이름. 1.x `Interpreter.getSignatureOutputs(key)`로 읽은 시그니처 출력 키다 — 모델
     * 변환기가 raw 텐서 이름과 다르게 지어 붙이는 경우가 있어서, 같은 .tflite 파일이라도
     * 두 API가 보는 "출력 이름"이 다를 수 있다. 시그니처가 없는 모델이면 raw 이름으로
     * 대체한다(마지막 수단).
     */
    var outputSignatureName: String = ""
        private set

    /** [outputSignatureName]과 함께 넘길 시그니처 키. 시그니처가 없으면 빈 문자열(기본 시그니처). */
    var outputSignatureKey: String = ""
        private set

    fun initialize() {
        require(modelFile.exists()) { "모델 파일 없음: ${modelFile.absolutePath}" }

        val options = Interpreter.Options().apply {
            setNumThreads(numThreads)
            xnnpackOverride?.let { setUseXNNPACK(it) }
        }
        val itp = Interpreter(modelFile, options)
        interpreter = itp

        signature = describeSignature(itp)
        bindTensors(itp)
        inputIs64Bit = BooleanArray(itp.inputTensorCount) { itp.getInputTensor(it).dataType() == DataType.INT64 }
        outputShape = itp.getOutputTensor(embeddingOutputIndex).shape()
        outputElements = itp.getOutputTensor(embeddingOutputIndex).numElements()

        if (backend == Backend.COMPILED_CPU) {
            // 시그니처용 Interpreter를 먼저 닫는다 — 둘이 동시에 떠 있으면 로드 직후 PSS가 +366MB까지 올랐다(Fold4, +242MB 대비).
            itp.close()
            interpreter = null
            val created = runCatching {
                CompiledModel.create(
                    modelFile.absolutePath,
                    CompiledModel.Options(Accelerator.CPU).apply { cpuOptions = CompiledModel.CpuOptions(numThreads, null, null) },
                )
            }.getOrNull()
            if (created != null) {
                compiled = created
                compiledInputs = created.createInputBuffers()
                compiledOutputs = created.createOutputBuffers()
                activeBackend = Backend.COMPILED_CPU
            } else {
                interpreter = Interpreter(modelFile, options)
                activeBackend = Backend.INTERPRETER
            }
        }
    }

    /**
     * ── 시그니처를 읽어서 보고하는 이유 ─────────────────────
     * 이 프로젝트는 학습 템플릿 ↔ 런타임 파서 포맷 계약 불일치로 세 버전을 날린 적이 있다
     * (finetune/README.md). 임베딩 모델도 같은 함정이 있다 — 입력이 `input_ids` 하나인지
     * `attention_mask`/`position_ids`까지 받는지, 출력이 이미 풀링된 [1,768]인지 토큰별
     * [1,seq,768]인지가 체크포인트마다 다르고, 틀려도 **예외 없이 그냥 나쁜 벡터가 나온다**.
     * 그래서 추측해서 하드코딩하지 않고, 실제 텐서를 읽어 문자열로 남긴 뒤 그걸 근거로
     * 바인딩한다. 첫 실행의 이 출력이 다음 코드를 쓰는 근거가 된다.
     */
    private fun describeSignature(itp: Interpreter): String = buildString {
        appendLine("inputs: ${itp.inputTensorCount}")
        repeat(itp.inputTensorCount) { i ->
            val t = itp.getInputTensor(i)
            appendLine("  [$i] ${t.name()} ${t.shape().contentToString()} ${t.dataType()}")
        }
        appendLine("outputs: ${itp.outputTensorCount}")
        repeat(itp.outputTensorCount) { i ->
            val t = itp.getOutputTensor(i)
            appendLine("  [$i] ${t.name()} ${t.shape().contentToString()} ${t.dataType()}")
        }
        val sigs = runCatching { itp.signatureKeys.toList() }.getOrDefault(emptyList())
        append("signatureKeys: ${if (sigs.isEmpty()) "(none)" else sigs.joinToString()}")
    }

    private fun bindTensors(itp: Interpreter) {
        val roles = mutableMapOf<InputRole, Int>()
        repeat(itp.inputTensorCount) { i ->
            val name = itp.getInputTensor(i).name().lowercase()
            val role = when {
                name.contains("attention") || name.contains("mask") -> InputRole.ATTENTION_MASK
                name.contains("position") -> InputRole.POSITION_IDS
                name.contains("token_type") || name.contains("segment") -> InputRole.TOKEN_TYPE_IDS
                name.contains("input") || name.contains("ids") || name.contains("token") ->
                    InputRole.INPUT_IDS
                else -> null
            }
            // 이름이 안 잡히는 입력이 딱 하나 남으면 그게 토큰 ID다 — 변환기가 이름을
            // `serving_default_args_0` 같이 뭉개는 경우가 흔해서 위치로 구제한다.
            if (role != null && !roles.containsKey(role)) roles[role] = i
        }
        if (!roles.containsKey(InputRole.INPUT_IDS) && itp.inputTensorCount > 0) {
            val taken = roles.values.toSet()
            val leftover = (0 until itp.inputTensorCount).filterNot { it in taken }
            // litert-torch의 embedding_gemma 변환기는 (tokens, attention_mask) 순서로 받고 이름을
            // serving_default_args_0/1로 뭉갠다. 그 경우만 위치로 정한다(2026-10-14, 짧은 시퀀스 재변환판).
            if (leftover.size == 2 && !roles.containsKey(InputRole.ATTENTION_MASK)) {
                roles[InputRole.INPUT_IDS] = leftover[0]
                roles[InputRole.ATTENTION_MASK] = leftover[1]
            }
        }
        if (!roles.containsKey(InputRole.INPUT_IDS) && itp.inputTensorCount > 0) {
            val taken = roles.values.toSet()
            val leftover = (0 until itp.inputTensorCount).filterNot { it in taken }
            check(leftover.size == 1) {
                "토큰 ID 입력을 특정할 수 없다. 실제 시그니처를 보고 매핑을 고칠 것:\n$signature"
            }
            roles[InputRole.INPUT_IDS] = leftover.single()
        }
        inputIndexByRole = roles

        val idsShape = itp.getInputTensor(roles.getValue(InputRole.INPUT_IDS)).shape()
        sequenceLength = idsShape.last()
        check(sequenceLength > 1) {
            "시퀀스 길이를 못 읽었다(shape=${idsShape.contentToString()}).\n$signature"
        }

        // 출력이 여러 개면 마지막 축이 768인 것을 임베딩으로 본다.
        embeddingOutputIndex = (0 until itp.outputTensorCount).firstOrNull { i ->
            itp.getOutputTensor(i).shape().last() == EMBEDDING_DIM
        } ?: 0
        outputTensorName = itp.getOutputTensor(embeddingOutputIndex).name()

        val sigKeys = runCatching { itp.signatureKeys.toList() }.getOrDefault(emptyList())
        val firstSigKey = sigKeys.firstOrNull()
        val sigOutputs = firstSigKey?.let {
            runCatching { itp.getSignatureOutputs(it).toList() }.getOrDefault(emptyList())
        } ?: emptyList()
        if (firstSigKey != null && sigOutputs.size == 1) {
            outputSignatureKey = firstSigKey
            outputSignatureName = sigOutputs.single()
        } else {
            // 시그니처가 없거나(구형 변환) 출력이 여러 개라 위치로 못 고르면, raw 이름이
            // 최선의 추측이다 — CompiledModel 쪽에서도 실패할 수 있다는 뜻이고, 그건
            // 호출한 쪽이 조회 결과 문자열로 드러낸다.
            outputSignatureKey = ""
            outputSignatureName = outputTensorName
        }
    }

    /**
     * 텍스트 하나를 768차원 단위벡터로 만든다. `prefix`는 반드시 [TaskPrefix]에서 고를 것 —
     * 이유는 그쪽 주석 참고.
     */
    fun embed(text: String, prefix: String): FloatArray {
        val tok = checkNotNull(tokenizer) {
            "이 인스턴스는 tokenizer 없이 만들어졌다. GemmaBpeTokenizer를 넘기거나 " +
                "embedTokens()로 토큰 ID를 직접 줄 것."
        }
        return embedTokens(tok.encode(prefix + text))
    }

    /**
     * 이미 토큰화된 ID 배열로 임베딩한다. 패딩 자리를 뺀 전부가 유효 토큰이라고 본다 —
     * 호출자가 잘라 넣지 않는 한 마스크는 전부 1이다.
     */
    fun embedTokens(tokenIds: IntArray): FloatArray {
        val ids = tokenIds
        // 잘림은 조용히 넘어가면 안 된다 — 글이 길어지면 뒤쪽이 통째로 사라지는데
        // 벡터는 멀쩡해 보인다. 호출자가 청킹을 결정할 수 있도록 여기서 알린다.
        lastInputWasTruncated = ids.size > sequenceLength

        val used = minOf(ids.size, sequenceLength)
        compiled?.let { return embedCompiled(it, ids, used) }
        val itp = checkNotNull(interpreter) { "initialize()를 먼저 호출할 것" }
        val inputs = HashMap<Int, Any>()
        inputs[inputIndexByRole.getValue(InputRole.INPUT_IDS)] =
            intBuffer(itp, inputIndexByRole.getValue(InputRole.INPUT_IDS)) { i ->
                if (i < used) ids[i] else PAD_TOKEN_ID
            }
        inputIndexByRole[InputRole.ATTENTION_MASK]?.let { idx ->
            inputs[idx] = intBuffer(itp, idx) { i -> if (i < used) 1 else 0 }
        }
        inputIndexByRole[InputRole.POSITION_IDS]?.let { idx ->
            inputs[idx] = intBuffer(itp, idx) { i -> i }
        }
        inputIndexByRole[InputRole.TOKEN_TYPE_IDS]?.let { idx ->
            inputs[idx] = intBuffer(itp, idx) { 0 }
        }

        val outputTensor = itp.getOutputTensor(embeddingOutputIndex)
        val outputBuffer = ByteBuffer
            .allocateDirect(outputTensor.numElements() * FLOAT_BYTES)
            .order(ByteOrder.nativeOrder())
        val outputs = HashMap<Int, Any>().apply { put(embeddingOutputIndex, outputBuffer) }

        itp.runForMultipleInputsOutputs(
            Array(itp.inputTensorCount) { inputs[it] ?: emptyInputFor(itp, it) },
            outputs,
        )

        outputBuffer.rewind()
        val floats = FloatArray(outputTensor.numElements()) { outputBuffer.float }
        val pooled = pool(floats, outputTensor.shape(), used)
        return l2Normalize(pooled)
    }

    /**
     * 문서를 잘리지 않게 임베딩한다. 시퀀스 길이에 들어가면 한 번, 넘치면 [DocumentChunker]로 나눠 청크마다
     * `title: … | text:` 프리픽스와 BOS/EOS를 다시 붙여 여러 번 돈다. 검색 점수는 [DocumentChunker.maxScore].
     * 질의는 [embed]에 [TaskPrefix.QUERY]로 한 번 넣으면 된다.
     *
     * [lastInputWasTruncated]를 보고 나서 다시 자르는 대신 토큰 수를 먼저 센다 — 같은 판정을 임베딩 한 번
     * 덜 하고 내린다.
     */
    fun embedDocument(body: String, title: String? = null): DocumentEmbedding {
        val tok = checkNotNull(tokenizer) { "embedDocument는 tokenizer가 필요하다" }
        check(sequenceLength > 0) { "initialize()를 먼저 호출할 것" }
        val t0 = System.nanoTime()
        val plan = DocumentChunker.plan(tok, title, body, sequenceLength)
        val tokenizeMs = (System.nanoTime() - t0) / 1e6
        val chunkMs = DoubleArray(plan.chunks.size)
        val vectors = plan.chunks.mapIndexed { i, ids ->
            val s = System.nanoTime()
            val v = embedTokens(ids)
            chunkMs[i] = (System.nanoTime() - s) / 1e6
            v
        }
        return DocumentEmbedding(vectors, plan.chunks, tokenizeMs, chunkMs, plan.titleMovedToBody)
    }

    class DocumentEmbedding(
        val chunkVectors: List<FloatArray>,
        val chunkTokenIds: List<IntArray>,
        val tokenizeMs: Double,
        val chunkMs: DoubleArray,
        val titleMovedToBody: Boolean,
    ) {
        fun score(query: FloatArray): Float = DocumentChunker.maxScore(query, chunkVectors)
    }

    /**
     * 여러 입력을 [batchSize]개씩 임베딩한다. [pacer]를 주면 호출마다 시간을 기록하고 배치 사이에서
     * [AdaptivePacer.afterBatch]가 필요하면 쉰다. [onItem]은 (순번, 벡터, 이 호출의 ms)로 불린다.
     */
    fun embedBatch(
        inputs: List<IntArray>,
        batchSize: Int = 16,
        pacer: AdaptivePacer? = null,
        onItem: (index: Int, vector: FloatArray, ms: Double) -> Unit = { _, _, _ -> },
    ) {
        inputs.forEachIndexed { i, ids ->
            val t0 = System.nanoTime()
            val v = embedTokens(ids)
            val ms = (System.nanoTime() - t0) / 1e6
            pacer?.record(ms)
            onItem(i, v, ms)
            if ((i + 1) % batchSize == 0 && i + 1 < inputs.size) pacer?.afterBatch()
        }
    }

    private fun embedCompiled(m: CompiledModel, ids: IntArray, used: Int): FloatArray {
        fun fill(index: Int, value: (Int) -> Int) {
            val buf = compiledInputs[index]
            if (inputIs64Bit[index]) buf.writeLong(LongArray(sequenceLength) { value(it).toLong() })
            else buf.writeInt(IntArray(sequenceLength) { value(it) })
        }
        for (i in compiledInputs.indices) {
            when (i) {
                inputIndexByRole[InputRole.INPUT_IDS] -> fill(i) { if (it < used) ids[it] else PAD_TOKEN_ID }
                inputIndexByRole[InputRole.ATTENTION_MASK] -> fill(i) { if (it < used) 1 else 0 }
                inputIndexByRole[InputRole.POSITION_IDS] -> fill(i) { it }
                else -> fill(i) { 0 }
            }
        }
        m.run(compiledInputs, compiledOutputs)
        val raw = compiledOutputs[embeddingOutputIndex].readFloat()
        return l2Normalize(pool(raw.copyOf(outputElements), outputShape, used))
    }

    /** 마지막 [embed] 호출에서 입력이 시퀀스 길이를 넘겨 잘렸는지. */
    var lastInputWasTruncated: Boolean = false
        private set

    /**
     * 출력이 이미 [1,768]이면 그대로, 토큰별 [1,seq,768]이면 마스크 평균을 낸다.
     * 마스크를 무시하고 전체 평균을 내면 패딩 토큰이 벡터를 문장 길이 쪽으로 끌고 가서,
     * 짧은 기록과 긴 기록이 내용과 무관하게 서로 비슷해진다.
     */
    private fun pool(flat: FloatArray, shape: IntArray, used: Int): FloatArray {
        if (shape.size <= 2) return flat.copyOf(EMBEDDING_DIM)

        val seq = shape[shape.size - 2]
        val dim = shape[shape.size - 1]
        val sums = FloatArray(dim)
        var counted = 0
        for (t in 0 until seq) {
            if (t >= used) continue
            counted++
            val base = t * dim
            for (d in 0 until dim) sums[d] += flat[base + d]
        }
        if (counted == 0) return sums
        for (d in 0 until dim) sums[d] /= counted
        return sums
    }

    private fun intBuffer(itp: Interpreter, index: Int, value: (Int) -> Int): ByteBuffer {
        val tensor = itp.getInputTensor(index)
        val elements = tensor.numElements()
        val is64Bit = tensor.dataType() == DataType.INT64
        val width = if (is64Bit) LONG_BYTES else INT_BYTES
        val buffer = ByteBuffer.allocateDirect(elements * width).order(ByteOrder.nativeOrder())
        for (i in 0 until elements) {
            val v = value(i)
            if (is64Bit) buffer.putLong(v.toLong()) else buffer.putInt(v)
        }
        buffer.rewind()
        return buffer
    }

    // 예상 못 한 추가 입력이 있어도 실행은 되게 0으로 채워 넣는다. 그런 입력이 존재한다는
    // 사실 자체는 signature에 남으므로 프로브 출력에서 드러난다.
    private fun emptyInputFor(itp: Interpreter, index: Int): ByteBuffer =
        intBuffer(itp, index) { 0 }

    private fun l2Normalize(v: FloatArray): FloatArray {
        var sum = 0.0
        for (x in v) sum += x.toDouble() * x
        val norm = kotlin.math.sqrt(sum).toFloat()
        if (norm == 0f) return v
        return FloatArray(v.size) { v[it] / norm }
    }

    override fun close() {
        interpreter?.close()
        interpreter = null
        compiledInputs.forEach { runCatching { it.close() } }
        compiledOutputs.forEach { runCatching { it.close() } }
        compiledInputs = emptyList(); compiledOutputs = emptyList()
        runCatching { compiled?.close() }
        compiled = null
    }

    private enum class InputRole { INPUT_IDS, ATTENTION_MASK, POSITION_IDS, TOKEN_TYPE_IDS }

    /**
     * ── 프리픽스를 틀리면 조용히 나빠진다 ────────────────────
     * EmbeddingGemma는 작업별 프리픽스를 붙여 학습됐고, 검색은 **비대칭**이다 — 질의와 문서에
     * 서로 다른 프리픽스를 써야 한다. 양쪽에 같은 걸 쓰거나 아예 안 붙여도 예외는 안 나고
     * 그럴듯한 벡터가 나오지만 검색 품질만 떨어진다. 그래서 [embed]가 프리픽스를 필수 인자로
     * 받는다 — 기본값을 두면 실수로 빼먹은 걸 알 방법이 없다.
     */
    object TaskPrefix {
        /** 검색 질의 쪽. 문서 쪽에는 [DOCUMENT]를 쓸 것. */
        const val QUERY = "task: search result | query: "

        /** 검색 대상 문서 쪽. 제목이 있으면 none 자리에 넣는다. */
        const val DOCUMENT = "title: none | text: "

        /** 문장끼리 얼마나 비슷한지 재는 대칭 비교용. 검색에는 쓰지 말 것. */
        const val SIMILARITY = "task: sentence similarity | query: "
    }

    companion object {
        const val EMBEDDING_DIM = 768

        /**
         * 프로브 실험용 스위치(`--ez xnnpack`). null이면 런타임 기본값. 2026-10-15 Fold4에서 스레드 수를 1~8로 바꿔도
         * 지연이 같아서(71.5ms) XNNPACK을 켜고 끈 경우를 비교하려고 둔다.
         */
        @Volatile var xnnpackOverride: Boolean? = null

        /** 프로브 실험용(`--es embedder compiled`). 생성자에서 backend를 안 주면 이 값을 쓴다. */
        @Volatile var backendOverride: Backend? = null

        private const val PAD_TOKEN_ID = 0
        private const val FLOAT_BYTES = 4
        private const val INT_BYTES = 4
        private const val LONG_BYTES = 8

        /** 두 단위벡터의 코사인 유사도. 정규화돼 있으므로 내적이면 충분하다. */
        fun cosine(a: FloatArray, b: FloatArray): Float {
            require(a.size == b.size) { "차원 불일치: ${a.size} vs ${b.size}" }
            var dot = 0.0
            for (i in a.indices) dot += a[i].toDouble() * b[i]
            return dot.toFloat()
        }
    }
}
