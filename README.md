# 🌍 **FaceID en el Aula: IA para Identificación Facial en Tiempo Real**

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-green.svg)](https://www.python.org/downloads/)
[![TensorFlow Lite](https://img.shields.io/badge/TensorFlow-Lite-yellow.svg)](https://www.tensorflow.org/lite)
[![Android 8+](https://img.shields.io/badge/android-8%2B-blue.svg)](https://developer.android.com/)
[![UAC Cusco](https://img.shields.io/badge/UAC-Cusco-orange.svg)](https://uacusco.edu.pe/)

> Proyecto académico integrando **Visión Computacional, Deep Learning y Computación Móvil**: demuestra cómo un modelo CNN puede ser desplegado eficientemente en Android usando TensorFlow Lite.
> Implementado bajo el marco **CRISP-ML** y optimizado con **TensorFlow Lite**.

---

## 🖼️ Imágenes de Referencia

<p align="center">
  <img src="docs/img/diagrama_crispml.png" width="350" alt="Flujo CRISP-ML">
  <img src="docs/img/demo_mobile.gif" width="200" alt="Demo Android">
  <img src="docs/img/resultados.png" width="350" alt="Resultados">
</p>

*(Puedes cambiar estas rutas o sustituir por tus propias capturas: diagrama, demo y resultados.)*

---

## 🧭 Descripción General

**FaceID en el Aula** es un sistema de identificación facial diseñado para reconocer miembros de un grupo utilizando únicamente la cámara de un móvil Android.  
El modelo emplea **MobileNetV2**, ajustado (*fine-tuning*) sobre un dataset personalizado y optimizado mediante **cuantización (float16/INT8)** para un rendimiento móvil sin conexión.

💡 El objetivo es demostrar todo el ciclo de *Machine Learning Engineering* — recolección de datos, entrenamiento, optimización y despliegue — siguiendo buenas prácticas de ingeniería.

---

## 🧠 Componentes Principales

| Etapa                | Herramienta/Librería           | Propósito                                   |
|----------------------|-------------------------------|---------------------------------------------|
| **Captura de Datos** | OpenCV                        | Recolección de imágenes faciales            |
| **Preprocesamiento** | Haar Cascade / Dlib           | Detección y recorte automático de rostros   |
| **Modelado CNN**     | TensorFlow + Keras            | Entrenamiento/transfer learning             |
| **Optimización**     | TensorFlow Lite               | Conversión y cuantización para móviles      |
| **App Móvil**        | Kotlin + CameraX + TFLite     | Inferencia en tiempo real                   |
| **Visualización**    | GitHub Pages + Chart.js       | Presentación interactiva de métricas        |

---

## ⚙️ Requisitos del Entorno

```bash
# Crear entorno virtual (Linux/macOS)
python -m venv .venv && source .venv/bin/activate
# En Windows:
.venv\Scripts\activate

pip install --upgrade pip
pip install opencv-python-headless tensorflow==2.16.1 tensorflow-model-optimization \
            scikit-learn matplotlib pandas jupyter onnx onnxruntime tflite-support
```

- Python ≥ 3.10
- GPU CUDA (opcional para acelerar entrenamiento)
- Android Studio Iguana o superior
- Dataset ≥ 500 imágenes totales (mínimo 100 por persona)

---

## 🧩 Estructura del Proyecto

```
faceid-aula/
├─ 1_data_collection/        # Captura de imágenes
├─ 2_data_prep/              # Preprocesamiento y partición
├─ 3_model/                  # Entrenamiento/exportación del modelo
├─ 4_mobile_app_android/     # App Android (CameraX + TFLite)
├─ models/                   # Modelos .h5 / .tflite optimizados
├─ data/                     # Datos (train/val/test)
└─ docs/                     # Infografía web (GitHub Pages)
```

---

## 🔁 Pipeline de Ejecución

```bash
# 1️⃣ Captura de rostros (mínimo 100 imágenes/persona)
python 1_data_collection/capture_opencv.py --person PersonaA --n 150

# 2️⃣ Recorte, normalización y división del dataset
python 2_data_prep/detect_crop.py
python 2_data_prep/split_dataset.py

# 3️⃣ Entrenamiento, evaluación y exportación TFLite
python 3_model/train_mobilenetv2.py
python 3_model/eval_report.py
python 3_model/tflite_convert.py
```

Se generan:
- Matriz de confusión (`models/confusion_matrix.csv`)
- Reporte de clasificación (Precision, Recall, F1)
- Modelos finales (.h5 y .tflite)

---

## 📊 Métricas Clave

| Indicador         | Objetivo          | Descripción                           |
|-------------------|------------------|---------------------------------------|
| Accuracy (Test)   | ≥ 90%            | Precisión general del modelo          |
| F1-Score          | ≥ 0.90           | Balance entre precisión y exhaustividad |
| Latencia Android  | ≤ 500 ms/frame   | Tiempo promedio de inferencia         |
| Tamaño modelo     | ≤ 20 MB          | Ideal para ejecución local            |
| FPS               | ≥ 10             | Fluidez aceptable en móviles          |

Con MobileNetV2 y un dataset balanceado, se alcanzan accuracies del 94–97% (5 clases).

---

## 📱 Despliegue Móvil (Android)

**Características:**
- Interfaz ligera con CameraX
- Procesamiento 100% local, sin conexión
- Etiquetado en pantalla por rostro con probabilidad
- Medición automática de latencia por frame
- Registro de logs en SQLite o Firebase Local

**Tecnologías:**
- Kotlin (nativo)
- TensorFlow Lite Interpreter
- Modelo: `faceid_best_float16.tflite`
- Compatibilidad: Android 8.0 (API 24)+

**Configuración:**
1. Abre la carpeta del proyecto Android en Android Studio:
   ```
   4_mobile_app_android/Android/
   ```
2. Sigue las instrucciones en el archivo `README_android_full.md`.

---

## 🧬 Innovaciones Técnicas

✅ Cuantización híbrida: float16 e INT8 para máxima eficiencia  
✅ Data Augmentation inteligente: rotación, brillo y simetría aleatoria  
✅ Explicabilidad (XAI): mapas Grad-CAM para visualización de decisión  
✅ Integración Edge TPU: compatibilidad con Coral USB Accelerator y RPi4  
✅ Seguridad biométrica local: embeddings cifrados (AES-256)  
✅ Inferencia híbrida: soporte opcional en servidor Flask o API REST

---

## 🧠 Futuras Mejoras

- 📸 Detección multi-rostro con bounding boxes dinámicos
- 🔊 Integración con VoiceID (reconocimiento de voz)
- 🧩 Reducción de sesgo de dataset (normalización de tono/fondo)
- ☁️ Sincronización en la nube (Firebase + almacenamiento privado)
- 🧠 Migración a Vision Transformers (ViT) o EfficientNet-Lite

---

## 🌈 Infografía Interactiva (GitHub Pages)

Incluye:
- Presentación del problema y objetivos
- Diagrama del flujo CRISP-ML
- Arquitectura MobileNetV2 y capas entrenadas
- Curvas de entrenamiento/evaluación
- Resultados visuales y vídeo demostrativo

Archivo principal: `docs/index.html`  
Publicación automática en GitHub Pages tras commit en la rama `main`.

---

## 📚 Referencias Técnicas

- Sandler, M. et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks.
- [TensorFlow Lite Docs](https://www.tensorflow.org/lite)
- CRISP-ML(Q): Springer (2021)
- [Android CameraX API](https://developer.android.com/training/camerax)

---

## 🧑‍💻 Autores

**Proyecto:** FaceID en el Aula  
**Facultad:** Ingeniería de Sistemas – UAC, Cusco  
**Versión:** 2.2 (2025)  
**Licencia:** MIT (uso académico y educativo)

🧾 *"Una IA responsable no reemplaza la mirada humana; la amplifica para crear conocimiento y seguridad en su entorno."*

---

Este texto está listo para **copiar y pegar directamente en tu repositorio GitHub** como `README.md` o en la infografía interactiva.  
Incluye estilo visual, estructura profesional y secciones innovadoras.
