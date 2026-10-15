package io.github.kakasolg.phototextsearch

/**
 * Inserts a short cooldown between batches when recent inference time drifts above the cold baseline.
 *
 * ── 왜 필요한가 (2026-10-14, Pixel 11 Pro XL) ────────────────────
 *  · 연속 임베딩 중 큰 코어 주파수 상한이 4.1GHz → 1.8~2.8GHz로 내려가 seq64가 29.8 → 71.1ms가 됐는데,
 *    그동안 Thermal Status는 대부분 0이었다. 그래서 열 상태가 아니라 **실제 추론 시간**을 신호로 쓴다.
 *  · 기준선은 워밍업을 뺀 첫 [baselineCalls]회의 중앙값, 비교 대상은 최근 [windowCalls]회 평균이다.
 *    평균이 기준선의 [slowdownRatio]배를 넘으면 배치가 끝날 때마다 [cooldownMs]만큼 쉰다.
 *  · **세션 안에서 잰 기준선만 쓰면 안 된다.** 이미 상한이 내려간 상태로 시작한 실행은 느린 속도를 기준선으로
 *    삼아 끝까지 "안 느려졌다"고 판단했다(기준선 41.5ms, 2,000건 중 쉼 1회). 그래서 차가운 기기에서 잰 값을
 *    [referenceBaselineMs]로 주면 세션 기준선과 둘 중 작은 쪽을 쓴다(seq64 Pixel 11 Pro XL 차가운 상태 ≈ 30ms).
 *  · 쉬는 것은 처리량을 위한 게 아니라 **건당 지연과 발열을 낮추기 위한 것**이다. 총 소요 시간이 줄어든다는
 *    보장은 없다.
 *
 * Not thread-safe; one pacer per indexing loop.
 */
class AdaptivePacer(
    private val warmupCalls: Int = 3,
    private val baselineCalls: Int = 10,
    private val windowCalls: Int = 10,
    private val slowdownRatio: Double = 1.3,
    private val cooldownMs: Long = 500,
    private val referenceBaselineMs: Double? = null,
    private val sleeper: (Long) -> Unit = { Thread.sleep(it) },
) {
    private var seen = 0
    private val baselineSamples = ArrayList<Double>(baselineCalls)
    private val window = ArrayDeque<Double>(windowCalls)

    /** Median of the first [baselineCalls] post-warmup calls; null until collected. */
    var sessionBaselineMs: Double? = null
        private set

    /** The baseline used for decisions: the smaller of the session baseline and [referenceBaselineMs]. */
    val baselineMs: Double?
        get() = listOfNotNull(sessionBaselineMs, referenceBaselineMs).minOrNull()

    var cooldowns = 0
        private set
    var totalSleptMs = 0L
        private set

    fun record(ms: Double) {
        seen++
        if (seen <= warmupCalls) return
        if (sessionBaselineMs == null) {
            baselineSamples += ms
            if (baselineSamples.size == baselineCalls) sessionBaselineMs = baselineSamples.sorted()[baselineCalls / 2]
        }
        window.addLast(ms)
        if (window.size > windowCalls) window.removeFirst()
    }

    fun movingAverageMs(): Double? = if (window.size < windowCalls) null else window.average()

    fun isSlow(): Boolean {
        val base = baselineMs ?: return false
        val ma = movingAverageMs() ?: return false
        return ma > base * slowdownRatio
    }

    /** Call at a batch boundary. Sleeps when slow and returns the time slept (0 otherwise). */
    fun afterBatch(): Long {
        if (!isSlow()) return 0
        sleeper(cooldownMs)
        cooldowns++
        totalSleptMs += cooldownMs
        return cooldownMs
    }
}
