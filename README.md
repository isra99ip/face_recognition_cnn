# FaceID en el Aula: Reconocimiento Facial con CNN + Despliegue Móvil

Pipeline: MobileNetV2 fine-tuning → TensorFlow Lite (float16/INT8) → App Android con CameraX + TFLite.

## Requisitos
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install opencv-python-headless tensorflow==2.16.1 tensorflow-model-optimization \
            scikit-learn matplotlib pandas jupyter onnx onnxruntime tflite-support
```

## Estructura
```
faceid-aula/
├─ 1_data_collection/
├─ 2_data_prep/
├─ 3_model/
├─ 4_mobile_app_android/
├─ models/
├─ data/
└─ docs/   # GitHub Pages
```

## Flujo rápido
```bash
# 1) Captura de imágenes
python 1_data_collection/capture_opencv.py --person PersonaA --n 150

# 2) Detección/recorte y split
python 2_data_prep/detect_crop.py
python 2_data_prep/split_dataset.py

# 3) Entrenamiento, evaluación y conversión a TFLite
python 3_model/train_mobilenetv2.py
python 3_model/eval_report.py
python 3_model/tflite_convert.py
```

## Métricas
- Objetivo: accuracy test ≥ 90% con 5 clases.
- Latencia objetivo en Android: < 500 ms por frame con TFLite float16.

## Android
Ver `4_mobile_app_android/README_android.md`.


## App Android completa
Abre `4_mobile_app_android/Android` en Android Studio y sigue `README_android_full.md`.
