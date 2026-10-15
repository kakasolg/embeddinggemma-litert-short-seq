package io.github.kakasolg.phototextsearch

import android.graphics.BitmapFactory
import android.graphics.Rect
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import java.io.File
import org.json.JSONArray
import org.json.JSONObject

/**
 * Photo/screenshot text demo, step 2: OCR every demo image on the device (README, "How it fails" §0).
 *
 * Same recognizer and decode as Field Lab's `MlKitPhotoSignalExtractor.readText` (ML Kit Korean recognizer, which
 * also reads Latin, at the original resolution). The demo images are all ≤ 2,560 px on the long edge, so Field Lab
 * would read each once without its 2×2 tile pass — this does exactly that.
 *
 * Input: a list file of image paths (the app can read files under /data/local/tmp but cannot list the directory).
 * Output: `demo_ocr.json` in [outDir] — per image the full text (Field Lab parity), [rows], block count and ms.
 */
object DemoOcrBench {

    /**
     * ── 왜 줄을 다시 묶나 ──────────────────────────────
     *  · ML Kit은 메뉴·영수증의 이름 칸과 가격 칸을 서로 다른 블록으로 돌려준다. result.text를 그대로 쓰면
     *    "김치찌개 … (메뉴 전체) … $16.99"처럼 이름과 가격이 수십 줄 떨어져, 검색·답이 짝을 잃는다.
     *  · 그래서 줄 상자를 왼쪽부터 보며, 바로 왼쪽에 끝나는 줄과 세로 중심이 줄 높이의 45% 안이면 같은 행으로
     *    잇는다. 행 전체가 아니라 마지막 줄과 비교하고, 사진 전체의 기울기(긴 줄들의 ML Kit Line.angle 중앙값)로
     *    오른쪽 높이를 내다보므로, 영수증처럼 이름과 가격 사이가 넓은 사진이 ±3° 기울어도 행이 끊기지 않는다.
     *  · 짧은 줄 하나의 각도는 흔들려서, 줄마다 각도를 쓰면 영수증의 Tax/Tip 가격이 옆 행에 붙었다(틀린 짝은
     *    짝이 없는 것보다 나쁘다). 그래서 사진 하나에 기울기 하나만 쓴다.
     */
    fun rows(result: Text): List<String> {
        val lines = result.textBlocks.flatMap { it.lines }.mapNotNull { l -> l.boundingBox?.let { OcrLine(it, l.angle, l.text) } }
            .sortedBy { it.box.left }
        val longAngles = lines.filter { it.box.width() >= it.box.height() * 4 }.map { it.angle }.sorted()
        val skew = if (longAngles.isEmpty()) 0f else longAngles[longAngles.size / 2]
        val slope = kotlin.math.tan(Math.toRadians(skew.toDouble())).toFloat()
        val rows = mutableListOf<MutableList<OcrLine>>()
        for (line in lines) {
            val box = line.box
            val tol = box.height() * 0.45f
            fun miss(row: List<OcrLine>): Float {
                val last = row.last()
                return kotlin.math.abs(last.box.exactCenterY() + slope * (box.exactCenterX() - last.box.exactCenterX()) - box.exactCenterY())
            }
            val target = rows.filter { it.last().box.right <= box.left + box.height() && miss(it) <= tol }.minByOrNull(::miss)
            if (target != null) target += line else rows += mutableListOf(line)
        }
        return rows.sortedBy { it.first().box.exactCenterY() }.map { row -> row.joinToString(" ") { it.text } }
    }

    private class OcrLine(val box: Rect, val angle: Float, val text: String)

    fun run(listPath: String, outDir: File): String {
        val paths = File(listPath).readLines().map { it.trim() }.filter { it.isNotEmpty() }
        val recognizer = TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build())
        val out = JSONArray()
        var totalMs = 0.0
        try {
            for (p in paths) {
                val bmp = BitmapFactory.decodeFile(p) ?: run {
                    out.put(JSONObject().put("path", p).put("error", "decode failed")); null
                } ?: continue
                try {
                    val t0 = System.nanoTime()
                    val result = Tasks.await(recognizer.process(InputImage.fromBitmap(bmp, 0)))
                    val ms = (System.nanoTime() - t0) / 1e6
                    totalMs += ms
                    val lines = JSONArray()
                    result.textBlocks.forEach { b -> b.lines.forEach { lines.put(it.text) } }
                    out.put(JSONObject().apply {
                        put("id", File(p).nameWithoutExtension); put("path", p)
                        put("w", bmp.width); put("h", bmp.height)
                        put("ms", ms); put("blocks", result.textBlocks.size)
                        put("text", result.text); put("lines", lines)
                        put("rows", JSONArray(rows(result)))
                    })
                } finally {
                    bmp.recycle()
                }
            }
        } finally {
            recognizer.close()
        }
        outDir.mkdirs()
        File(outDir, "demo_ocr.json").writeText(JSONObject().put("device", android.os.Build.MODEL).put("items", out).toString(1))
        return "ocr done: ${paths.size}장, 합계 ${"%.0f".format(totalMs)}ms"
    }
}
