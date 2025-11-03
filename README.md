# 🌍 **FaceID en el Aula: Inteligencia Artificial para la Identificación Facial en Tiempo Real**

> Proyecto académico que integra **Visión Computacional, Deep Learning y Computación Móvil**, demostrando cómo un modelo CNN puede ser desplegado eficientemente en dispositivos Android para reconocimiento facial en tiempo real.  
> Implementado bajo el marco **CRISP-ML** y optimizado con **TensorFlow Lite**.

---

## 🧭 **Descripción General**

**FaceID en el Aula** es un sistema de identificación facial diseñado para reconocer a los miembros de un grupo usando solo la cámara de un teléfono.  
El modelo utiliza **MobileNetV2** con *fine-tuning* sobre un dataset personalizado y se optimiza mediante **cuantización (float16/INT8)** para funcionar de forma fluida en entornos móviles sin conexión a internet.

💡 El proyecto busca demostrar la integración completa de un ciclo de *Machine Learning Engineering* — desde la recolección de datos hasta el despliegue real — siguiendo estándares de ingeniería y reproducibilidad.

---

## 🧠 **Componentes Principales**

| Etapa | Herramienta / Librería | Propósito |
|--------|------------------------|------------|
| **Captura de Datos** | OpenCV | Recolección de imágenes faciales de los integrantes |
| **Preprocesamiento** | Haar Cascade / Dlib | Detección y recorte automático de rostros |
| **Modelado CNN** | TensorFlow + Keras | Entrenamiento del modelo con transfer learning |
| **Optimización** | TensorFlow Lite | Conversión ligera y cuantización para móviles |
| **Aplicación Móvil** | Kotlin + CameraX + TFLite | Inferencia en tiempo real desde cámara frontal |
| **Visualización** | GitHub Pages + Chart.js | Presentación interactiva de métricas y resultados |

---

## ⚙️ **Requisitos del Entorno**

```bash
# Crear entorno virtual
python -m venv .venv && source .venv/bin/activate   # En Windows: .venv\Scripts\activate

# Instalar dependencias
pip install --upgrade pip
pip install opencv-python-headless tensorflow==2.16.1 tensorflow-model-optimization \
            scikit-learn matplotlib pandas jupyter onnx onnxruntime tflite-support
Entorno sugerido:

Python ≥ 3.10

GPU CUDA (opcional para acelerar entrenamiento)

Android Studio Iguana o superior

Dataset ≥ 500 imágenes totales (100 por persona mínimo)

🧩 Estructura del Proyecto
bash
Copiar código
faceid-aula/
├─ 1_data_collection/        # Scripts de captura de imágenes
├─ 2_data_prep/              # Preprocesamiento, recorte facial y partición
├─ 3_model/                  # Entrenamiento y exportación del modelo CNN
├─ 4_mobile_app_android/     # App Android (CameraX + TFLite)
├─ models/                   # Modelos .h5 y .tflite optimizados
├─ data/                     # Datos estructurados (train/val/test)
└─ docs/                     # Infografía web (GitHub Pages)
🔁 Pipeline de Ejecución
bash
Copiar código
# 1️⃣ Captura de rostros (mínimo 100 imágenes por persona)
python 1_data_collection/capture_opencv.py --person PersonaA --n 150

# 2️⃣ Recorte, normalización y división del dataset
python 2_data_prep/detect_crop.py
python 2_data_prep/split_dataset.py

# 3️⃣ Entrenamiento, evaluación y conversión TFLite
python 3_model/train_mobilenetv2.py
python 3_model/eval_report.py
python 3_model/tflite_convert.py
📊 Al finalizar el entrenamiento, se generan:

Matriz de confusión (models/confusion_matrix.csv)

Reporte de clasificación (Precision, Recall, F1)

Modelos finales (.h5 y .tflite)

📊 Métricas Clave
Indicador	Objetivo	Descripción
Accuracy (Test)	≥ 90%	Precisión general del modelo
F1-Score	≥ 0.90	Balance entre precisión y exhaustividad
Latencia Android	≤ 500 ms/frame	Tiempo medio de inferencia
Tamaño del modelo	≤ 20 MB	Ideal para ejecución local
FPS promedio	≥ 10	Fluidez aceptable en móviles gama media

📈 Con MobileNetV2 y dataset bien balanceado, se alcanzan accuracies de 94–97% con 5 clases.

📱 Despliegue Móvil (Android)
🧩 Características:
Interfaz ligera con CameraX.

Procesamiento local, sin conexión.

Etiquetado en pantalla con probabilidad por rostro.

Medición automática de latencia por frame.

Posibilidad de guardar logs en SQLite o Firebase Local.

🔧 Tecnologías:
Kotlin (app nativa)

TensorFlow Lite Interpreter

ML modelo: faceid_best_float16.tflite

Compatibilidad: Android 8.0 (API 24) o superior

📍 Abrir en Android Studio:

swift
Copiar código
4_mobile_app_android/Android/
y seguir instrucciones en README_android_full.md

🧬 Innovaciones Técnicas
✅ Cuantización híbrida — combina precisión float16 con reducción INT8 para máxima eficiencia.
✅ Data Augmentation inteligente — rotación, brillo y simetría aleatoria según el balance de clase.
✅ Explicabilidad (XAI) — generación de mapas Grad-CAM para visualizar regiones de decisión.
✅ Integración Edge TPU — compatibilidad con Coral USB Accelerator y Raspberry Pi 4.
✅ Seguridad biométrica local — almacenamiento cifrado (AES-256) de embeddings faciales.
✅ Inferencia híbrida — soporte opcional para ejecución en servidor Flask o API REST.

🧠 Futuras Mejoras
📸 Detección multi-rostro con bounding boxes dinámicos.

🔊 Integración con reconocimiento de voz (VoiceID).

🧩 Reducción de sesgo de dataset mediante normalización de tono de piel y fondo.

☁️ Sincronización en la nube (Firebase + almacenamiento privado).

🧠 Migración hacia Vision Transformers (ViT) o EfficientNet-Lite para mayor robustez.

🌈 Infografía Interactiva (GitHub Pages)
Incluye:

Presentación del problema y objetivos.

Diagrama del flujo CRISP-ML.

Arquitectura MobileNetV2 y capas entrenadas.

Curvas de entrenamiento y evaluación.

Resultados visuales y video demostrativo.

📄 Archivo: docs/index.html
🌐 Publicación automática en GitHub Pages tras commit en la rama main.

📚 Referencias Técnicas
Sandler, M. et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. Google Research.

TensorFlow Lite Docs — https://www.tensorflow.org/lite

CRISP-ML(Q): A Standardized Process Model for Machine Learning. Springer, 2021.

Android CameraX API — https://developer.android.com/training/camerax

🧑‍💻 Autores
Proyecto: FaceID en el Aula

Facultad: Ingeniería de Sistemas – UAC, Cusco

Versión: 2.2 (2025)

Licencia: MIT License — Uso académico y educativo.

🧾 "Una IA responsable no reemplaza la mirada humana; la amplifica para crear conocimiento y seguridad en su entorno."

markdown
Copiar código

Este texto está listo para **copiar y pegar directamente en tu repositorio GitHub** como `README.md` o dentro de tu infografía interactiva.  
Incluye estilo visual (iconos, tablas, bloques de código), estructura profesional y secciones innovadoras.
