# App Android: CameraX + TensorFlow Lite

## Dependencias (build.gradle)
```
implementation "org.tensorflow:tensorflow-lite-task-vision:0.4.4"
implementation "androidx.camera:camera-camera2:1.3.4"
implementation "androidx.camera:camera-lifecycle:1.3.4"
implementation "androidx.camera:camera-view:1.3.4"
```

## Pasos
1. Copia el modelo `models/faceid_best_float16.tflite` a `app/src/main/ml/`.
2. Copia `3_model/labels.txt` a `app/src/main/assets/`.
3. Implementa el analizador de frames con CameraX. Convierte YUV→RGB, redimensiona a 224×224 y clasifica con TFLite Task Library.
4. Mide latencia con <code>System.nanoTime()</code> y muestra etiqueta + probabilidad.
