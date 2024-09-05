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
    }

    fun setupEmbedder() {
        val baseOptionsBuilder = BaseOptions.builder()
        try {
            when (currentDelegate) {
                DELEGATE_CPU -> {
                    baseOptionsBuilder.setDelegate(Delegate.CPU)
                }
                DELEGATE_GPU -> {
                    baseOptionsBuilder.setDelegate(Delegate.GPU)
                }
            }
        } catch (e: Exception) {
            Log.e(
                TAG, "2. Text embedder failed to load the model with error: " + e.message
            )
        }

        try {
            when (currentModel) {
                MODEL_MOBILE_BERT -> {
                    baseOptionsBuilder.setModelAssetPath(MODEL_MOBILE_BERT_PATH)
                }
                MODEL_AVERAGE_WORD -> {
                    baseOptionsBuilder.setModelAssetPath(MODEL_AVERAGE_WORD_PATH)
                }
                MODEL_UNIVERSAL_ENCODER -> {
                    baseOptionsBuilder.setModelAssetPath(MODEL_UNIVERSAL_ENCODER_PATH)
                }
            }
        } catch (e: Exception) {
            Log.e(
                TAG, "2. Text embedder failed to load the model with error: " + e.message
            )
        }


        try {
            val baseOptions = baseOptionsBuilder.build()
            val optionsBuilder = TextEmbedderOptions
                                    .builder()
                                    .setBaseOptions(baseOptions)
                                    .setQuantize(true)
                                    .setL2Normalize(true)
            val options = optionsBuilder.build()
            Log.i("CTX", context.toString())
            Log.i("CTX", options.toString())
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
        const val MODEL_MOBILE_BERT = 1
        const val MODEL_AVERAGE_WORD = 2
        const val MODEL_UNIVERSAL_ENCODER_PATH = "universal_sentence_encoder.tflite"
        const val MODEL_MOBILE_BERT_PATH = "mobile_bert.tflite"
        const val MODEL_AVERAGE_WORD_PATH = "average_word.tflite"
        const val OTHER_ERROR = 0
        const val GPU_ERROR = 1
        private const val TAG = "TextEmbedder"
    }

    interface EmbedderListener {
        fun onError(error: String, errorCode: Int = OTHER_ERROR)
    }
}