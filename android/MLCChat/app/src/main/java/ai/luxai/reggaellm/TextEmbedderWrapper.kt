package ai.luxai.reggaellm

import android.content.Context
import android.os.SystemClock
import android.util.Log

import com.google.mediapipe.tasks.components.containers.Embedding
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.text.textembedder.TextEmbedder
import com.google.mediapipe.tasks.text.textembedder.TextEmbedder.TextEmbedderOptions

class TextEmbedderWrapper(
    private val context: Context,
    private var currentDelegate: Int = DELEGATE_GPU,
    private var currentModel: Int = MODEL_UNIVERSAL_ENCODER,
    var listener: EmbedderListener? = null
) {
    private var textEmbedder: TextEmbedder? = null

    init {
        setupEmbedder()
    }

    fun setupEmbedder() {
        val baseOptionsBuilder = BaseOptions.builder()
        when (currentDelegate) {
            DELEGATE_CPU -> {
                baseOptionsBuilder.setDelegate(Delegate.CPU)
            }
            DELEGATE_GPU -> {
                baseOptionsBuilder.setDelegate(Delegate.GPU)
            }
        }
        when (currentModel) {
            MODEL_UNIVERSAL_ENCODER -> {
                baseOptionsBuilder.setModelAssetPath((MODEL_UNIVERSAL_ENCODER_PATH))
            }
        }
        try {
            val baseOptions = baseOptionsBuilder.build()
            val optionsBuilder = TextEmbedderOptions.builder().setBaseOptions(baseOptions)
            val options = optionsBuilder.build()
            textEmbedder = TextEmbedder.createFromOptions(context, options)
        } catch (e: IllegalStateException) {
            listener?.onError(
                "Text embedder failed to initialize. See error logs for details"
            )
            Log.e(
                TAG, "Text embedder failed to load the model with error: " + e.message
            )
        } catch (e: RuntimeException) {
            listener?.onError(
                "Text embedder failed to initialize. See error logs for details", GPU_ERROR
            )
            Log.e(
                TAG, "Text embedder failed to load model with error: " + e.message
            )
        }
    }

    fun embed (text: String): ResultBundle? {
        val startTime = SystemClock.uptimeMillis()

        textEmbedder?.let {
            val embed = it.embed(text).embeddingResult().embeddings().first()
            val inferenceTime = SystemClock.uptimeMillis() - startTime
            return ResultBundle(
                embed, inferenceTime
            )
        }
        return null
    }

    fun clearEmbedder() {
        textEmbedder?.close()
        textEmbedder = null
    }

    data class ResultBundle(
        val embedding:  Embedding,
        val inferenceTime: Long
    )

    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val MODEL_UNIVERSAL_ENCODER = 0
        const val MODEL_UNIVERSAL_ENCODER_PATH = "universal_encoder.tflite"
        const val OTHER_ERROR = 0
        const val GPU_ERROR = 1
        private const val TAG = "TextEmbedder"
    }

    interface EmbedderListener {
        fun onError(error: String, errorCode: Int = OTHER_ERROR)
    }
}