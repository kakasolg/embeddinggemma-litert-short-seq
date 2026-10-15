package io.github.kakasolg.phototextsearch

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier

/**
 * `adb shell am start -n io.github.kakasolg.phototextsearch/.MainActivity` opens the demo.
 * `--ez dev true` shows the prompt-version chips and the "60 questions" button.
 * `--es batch <tag> [--ez reindex true] [--ei delay 30]` runs the 60 fixed questions → files/photo_demo/run_<tag>.json.
 * Start measured runs with a tap, not from `am start`: an app launched by adb ran in a slower CPU group (see README).
 */
class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val dev = intent.getBooleanExtra("dev", false)
        val batch = intent.getStringExtra("batch")
        val reindex = intent.getBooleanExtra("reindex", false)
        val delay = intent.getIntExtra("delay", 0)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize().statusBarsPadding()) {
                    PhotoTextSearchDemoScreen(
                        onBack = { finish() },
                        batchTag = batch,
                        batchReindex = reindex,
                        batchDelaySec = delay,
                        devControls = dev,
                    )
                }
            }
        }
    }
}
