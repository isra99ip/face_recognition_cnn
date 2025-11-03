package com.example.faceid

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.io.BufferedReader
import java.io.InputStreamReader
import kotlin.math.min

class FaceIDInterpreter(ctx: Context, modelPath: String = "faceid_best_float16.tflite", labelsPath: String = "labels.txt") {
    private val interpreter: Interpreter
    private val labels: List<String>
    private val imgSize = 224
    private val nChannels = 3

    init {
        interpreter = Interpreter(loadModelFile(ctx, modelPath), Interpreter.Options().apply { numThreads = 2 })
        labels = loadLabels(ctx, labelsPath)
    }

    private fun loadModelFile(context: Context, modelPath: String): ByteBuffer {
        // Try ml/ then assets/
        val afd = try { context.assets.openFd(modelPath) } catch (e: Exception) { context.assets.openFd("ml/$modelPath") }
        val inputStream = FileInputStream(afd.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = afd.startOffset
        val declaredLength = afd.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun loadLabels(context: Context, labelsPath: String): List<String> {
        val candidates = listOf(labelsPath, "ml/$labelsPath")
        for (p in candidates) {
            try {
                val br = BufferedReader(InputStreamReader(context.assets.open(p)))
                return br.readLines().filter { it.isNotBlank() }
            } catch (_: Exception) { }
        }
        return emptyList()
    }

    private fun preprocess(bmp: Bitmap): ByteBuffer {
        val input = ByteBuffer.allocateDirect(4 * imgSize * imgSize * nChannels).order(ByteOrder.nativeOrder())
        val scaled = bmp.centerCropResize(imgSize)
        val pixels = IntArray(imgSize * imgSize)
        scaled.getPixels(pixels, 0, imgSize, 0, 0, imgSize, imgSize)
        var idx = 0
        for (y in 0 until imgSize) {
            for (x in 0 until imgSize) {
                val p = pixels[idx++]
                val r = ((p shr 16) and 0xFF) / 255.0f
                val g = ((p shr 8) and 0xFF) / 255.0f
                val b = (p and 0xFF) / 255.0f
                input.putFloat(r); input.putFloat(g); input.putFloat(b)
            }
        }
        return input
    }

    fun classify(bmp: Bitmap): Pair<String, Float> {
        val input = preprocess(bmp)
        // Output: 1 x N classes
        val numClasses = if (labels.isNotEmpty()) labels.size else 5
        val output = Array(1) { FloatArray(numClasses) }
        val t0 = System.nanoTime()
        interpreter.run(input, output)
        val dtMs = (System.nanoTime() - t0) / 1e6
        val probs = output[0]
        var bestIdx = 0
        var best = -1f
        for (i in probs.indices) {
            if (probs[i] > best) { best = probs[i]; bestIdx = i }
        }
        val label = if (bestIdx < labels.size) labels[bestIdx] else "Clase $bestIdx"
        return label to best
    }
}
