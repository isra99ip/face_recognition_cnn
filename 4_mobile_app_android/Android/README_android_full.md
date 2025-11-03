# Plantilla Android completa (CameraX + TFLite)

## Uso
1. Copia tu modelo `models/faceid_best_float16.tflite` del proyecto raíz a:
   - `app/src/main/ml/faceid_best_float16.tflite`
2. Copia `3_model/labels.txt` a:
   - `app/src/main/assets/labels.txt`
3. Abre la carpeta `4_mobile_app_android/Android` en Android Studio.
4. Sincroniza Gradle y ejecuta en un dispositivo físico.
5. La app usa la cámara frontal, hace center-crop + resize 224×224 y clasifica.

## Notas
- Esta plantilla usa el **Interpreter** directo para evitar requisitos de metadata.
- Si la latencia es alta, prueba: reducir entrada a 160, subir `numThreads`, o cuantizar INT8.
- Para mayor robustez añade un detector de rostro (ML Kit) antes de clasificar.
