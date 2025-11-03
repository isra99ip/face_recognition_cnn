package com.example.faceid

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.camera.view.PreviewView
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var txtResult: TextView
    private lateinit var txtLatency: TextView
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private var classifier: FaceIDInterpreter? = null

    private val requestPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) startCamera() else finish() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.previewView)
        txtResult = findViewById(R.id.txtResult)
        txtLatency = findViewById(R.id.txtLatency)

        classifier = FaceIDInterpreter(this)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            requestPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            val analyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            analyzer.setAnalyzer(cameraExecutor) { imageProxy ->
                try {
                    val bmp = imageProxy.toBitmap()
                    val t0 = System.nanoTime()
                    val (label, prob) = classifier?.classify(bmp) ?: ("N/A" to 0f)
                    val ms = ((System.nanoTime() - t0) / 1e6).toInt()
                    runOnUiThread {
                        txtResult.text = "$label  ${(prob*100).toInt()}%"
                        txtLatency.text = "$ms ms"
                    }
                } catch (_: Exception) {
                    // ignore frame errors
                } finally {
                    imageProxy.close()
                }
            }

            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, analyzer)
            } catch (_: Exception) { }
        }, ContextCompat.getMainExecutor(this))
    }
}
