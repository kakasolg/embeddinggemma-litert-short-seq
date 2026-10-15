package io.github.kakasolg.phototextsearch

/**
 * Store names read from the indexed photos' OCR rows, for the photo-text demo's v5 check
 * ([PhotoTextSearchDemo.PromptVersion.V5]). Same rules as demo/photo_text_search/store_check_dryrun.py.
 *
 * ── 가게 이름을 어떻게 뽑나 ──────────────────────────────
 *  · 사진마다 첫 내용 줄: 상태 표시줄(시각·5G) 줄은 빼고, 앱 화면이면("Details" 줄) 그 아래 화면 제목 한 줄도 건너뛴다.
 *  · 그 줄에서 영어 첫 두 단어(붙여서), 영어 한 단어뿐이면 5자 이상일 때 그 단어, 한글 첫 단어(2자 이상)를 이름 키로 쓴다.
 *    한 줄에 영어·한글이 같이 있으면("한빛식당 Hanbit Kitchen") 같은 가게로 묶는다.
 *  · 모든 비교는 소문자·영숫자·한글만 남긴 문자열의 포함 여부다.
 *
 * ── 알려진 한계 ──────────────────────────────
 *  · 첫 줄이 가게가 아닌 사진(날씨·할 일·"2 HOUR PARKING"·"주차장 이용 시간")도 키가 생긴다. 질문이 그 말을 쓰고 정답 사진이 다른 가게면
 *    맞는 답을 버릴 수 있다 — 버리는 조건에 "사진이 다른 아는 가게의 것"을 넣어 줄였지만 없앤 건 아니다.
 *  · 다른 언어 표기가 데이터 어디에도 같이 적혀 있지 않으면(Daon ↔ 다온) 같은 가게로 못 묶는다. 이때는 질문이 아는 가게를 부르지
 *    않은 것으로 보고 확인을 건너뛴다(버리지 않는 쪽).
 *  · 규칙(상태 표시줄·"Details" 아래 제목 건너뛰기)은 데모 사진의 첫 줄들을 보고 정했다. 일반적인 앱 화면 배치지만 이 데이터를 본 뒤의
 *    선택이라는 점은 그대로 적어 둔다.
 */
class PhotoStoreNames(val photos: List<PhotoTextSearchDemo.Photo>) {
    private val ownKeys: Map<String, Set<String>> = photos.associate { p -> p.id to (nameRow(p.rows)?.let(::keys) ?: emptySet()) }
    private val groupOf: Map<String, String>
    private val groups: Map<String, Set<String>>
    private val text: Map<String, String> = photos.associate { it.id to norm(it.rows.joinToString(" ")) }

    init {
        val parent = HashMap<String, String>()
        fun find(a: String): String {
            var x = a
            while (parent.getOrPut(x) { x } != x) x = parent.getValue(x)
            return x
        }
        ownKeys.values.forEach { ks ->
            val list = ks.toList()
            list.forEach { find(it) }
            list.drop(1).forEach { parent[find(it)] = find(list.first()) }
        }
        groupOf = parent.keys.associateWith { find(it) }
        groups = groupOf.entries.groupBy({ it.value }, { it.key }).mapValues { it.value.toSet() }
    }

    /** (keep?, reason). Drops only when the question names a known store the photo lacks and the photo is another known store. */
    fun keep(question: String, photoId: String): Pair<Boolean, String> {
        val q = norm(question)
        val mentioned = groups.filterValues { keys -> keys.any { it in q } }.keys
        if (mentioned.isEmpty()) return true to "질문에 아는 가게 없음"
        val t = text[photoId].orEmpty()
        val wanted = mentioned.flatMap { groups.getValue(it) }
        if (wanted.any { it in t }) return true to "사진에 그 가게 이름 있음"
        val own = ownKeys[photoId].orEmpty().mapNotNull { groupOf[it] }.toSet()
        if (own.isEmpty() || own.any { it in mentioned }) return true to "사진 가게 모름"
        return false to "질문 ${wanted.sorted()} / 사진 ${ownKeys[photoId].orEmpty().sorted()}"
    }

    companion object {
        private val STATUS = Regex("""^\s*\d{1,2}\s*:\s*\d{2}\b|\b5G\b""")
        private val NAV = Regex("""^[«<]?\s*Details$""")
        private val LETTER = Regex("[A-Za-z가-힣]")

        fun norm(s: String): String = s.lowercase().filter { it in 'a'..'z' || it in '0'..'9' || it in '가'..'힣' }

        fun nameRow(rows: List<String>): String? {
            var content = rows.filter { !STATUS.containsMatchIn(it) && LETTER.containsMatchIn(it) }
            val nav = content.indexOfFirst { NAV.matches(it.trim()) }
            if (nav >= 0) content = content.drop(nav + 2)
            return content.firstOrNull()
        }

        fun keys(row: String): Set<String> {
            val latin = Regex("[A-Za-z0-9]+").findAll(row).map { it.value.lowercase() }.toList()
            val hangul = Regex("[가-힣]+").findAll(row).map { it.value }.toList()
            val out = mutableSetOf<String>()
            if (latin.size >= 2) {
                out += latin[0] + latin[1]
            } else if (latin.size == 1 && latin[0].length >= 5 && latin[0].any { it in 'a'..'z' }) {
                out += latin[0]
            }
            hangul.firstOrNull()?.takeIf { it.length >= 2 }?.let { out += it }
            return out
        }
    }
}
