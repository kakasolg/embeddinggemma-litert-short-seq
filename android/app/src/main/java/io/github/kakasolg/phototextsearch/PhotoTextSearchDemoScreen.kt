package io.github.kakasolg.phototextsearch

import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel

/**
 * Three of the 60 fixed questions (C01, C04, A05) — each gave the same answer in every v3/v5 run on the Pixel:
 * a Korean question answered from an English sign, a negative answer, and a question with no answer in the photos.
 */
private val EXAMPLE_QUESTIONS = listOf(
    "메이플 플라자 주차 하루 최대 요금",
    "태국 식당 매운맛 5단계는 환불돼?",
    "Moonrise Donuts happy hour",
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PhotoTextSearchDemoScreen(
    onBack: () -> Unit,
    batchTag: String? = null,
    batchReindex: Boolean = false,
    batchDelaySec: Int = 0,
    devControls: Boolean = false,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current.applicationContext
    val viewModel: PhotoTextSearchDemoViewModel = viewModel(
        factory = object : ViewModelProvider.Factory {
            @Suppress("UNCHECKED_CAST")
            override fun <T : androidx.lifecycle.ViewModel> create(modelClass: Class<T>): T =
                PhotoTextSearchDemoViewModel(context) as T
        },
    )
    val status by viewModel.status.collectAsStateWithLifecycle()
    val busy by viewModel.busy.collectAsStateWithLifecycle()
    val query by viewModel.query.collectAsStateWithLifecycle()
    val offline by viewModel.offline.collectAsStateWithLifecycle()
    val promptVersion by viewModel.promptVersion.collectAsStateWithLifecycle()
    var input by rememberSaveable { mutableStateOf("") }

    LaunchedEffect(batchTag) {
        if (batchTag != null) viewModel.runBatch(batchTag, batchReindex, batchDelaySec) else viewModel.prepare()
    }

    Column(modifier = modifier.fillMaxSize()) {
        TopAppBar(
            title = { Text("사진 속 글자 찾기 (오프라인)") },
            navigationIcon = { TextButton(onClick = onBack) { Text("←") } },
        )
        Column(
            modifier = Modifier.fillMaxSize().padding(horizontal = 16.dp).verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(status, style = MaterialTheme.typography.bodySmall)
            Text(offline, style = MaterialTheme.typography.bodySmall)
            OutlinedTextField(
                value = input,
                onValueChange = { input = it },
                label = { Text("질문 (예: 메이플 플라자 몇 분 무료?)") },
                modifier = Modifier.fillMaxWidth(),
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.horizontalScroll(rememberScrollState())) {
                EXAMPLE_QUESTIONS.forEach { q ->
                    AssistChip(onClick = { input = q; viewModel.ask(q) }, label = { Text(q) }, enabled = !busy)
                }
            }
            if (devControls) Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("프롬프트", style = MaterialTheme.typography.bodySmall)
                PhotoTextSearchDemo.PromptVersion.entries.forEach { v ->
                    FilterChip(
                        selected = promptVersion == v,
                        onClick = { viewModel.setPromptVersion(v) },
                        label = { Text(v.name.lowercase()) },
                        enabled = !busy,
                    )
                }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = { viewModel.ask(input) }, enabled = !busy && input.isNotBlank()) { Text("찾기") }
                if (devControls) {
                    OutlinedButton(onClick = { viewModel.runBatch("manual", reindex = true, delaySec = 30) }, enabled = !busy) {
                        Text("60문항 측정")
                    }
                }
            }
            if (query.question.isNotBlank()) {
                Text("검색 ${"%.0f".format(query.searchMs)} ms", style = MaterialTheme.typography.labelSmall)
                val answer = query.answer
                when {
                    query.running && answer == null -> Text("Gemma가 읽는 중…")
                    answer != null && answer.cited.isNullOrEmpty() ->
                        Text("사진에서 답을 찾지 못했어요.", fontWeight = FontWeight.Bold)
                    answer != null -> Text(answer.text, fontWeight = FontWeight.Bold)
                }
                answer?.let {
                    Text(
                        "판단 ${"%.0f".format(it.gateMs)} ms · 답 ${"%.0f".format(it.answerMs)} ms",
                        style = MaterialTheme.typography.labelSmall,
                    )
                }
                query.hits.forEachIndexed { i, hit ->
                    val citedHere = answer?.cited?.contains(hit.photo.id) == true
                    HitCard(i + 1, hit, citedHere)
                }
            }
        }
    }
}

@Composable
private fun HitCard(rank: Int, hit: PhotoTextSearchDemo.Hit, cited: Boolean) {
    val thumb = remember(hit.photo.path) {
        BitmapFactory.decodeFile(hit.photo.path, BitmapFactory.Options().apply { inSampleSize = 8 })?.asImageBitmap()
    }
    Card(modifier = Modifier.fillMaxWidth()) {
        Row(modifier = Modifier.padding(8.dp), verticalAlignment = Alignment.CenterVertically) {
            thumb?.let {
                Image(it, contentDescription = null, contentScale = ContentScale.Crop, modifier = Modifier.size(72.dp))
            }
            Column(modifier = Modifier.padding(start = 8.dp)) {
                Text(
                    "사진$rank · ${"%.3f".format(hit.score)}${if (cited) " · 근거" else ""}",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = if (cited) FontWeight.Bold else FontWeight.Normal,
                )
                Text(hit.photo.rows.take(3).joinToString(" / "), style = MaterialTheme.typography.bodySmall, maxLines = 2)
            }
        }
    }
}
