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
# FaceID en el Aula: Informe Técnico y Flujo Completo

---

## 1. 🏫 Comprensión del Negocio

### 🎯 Objetivo
- Identificar integrantes (3–5 identidades) en tiempo real usando la cámara de un dispositivo Android **sin conexión a internet**.

### 📦 Alcance
- Reconocimiento de **una persona dominante** por frame, modo académico/prototipo.
- Quedan fuera: detección multi-persona en paralelo, verificación 1:1, anti-spoofing, público abierto.

---

## 2. ⚙️ Requerimientos Funcionales

- Captura de video con CameraX (`15–30 fps`, **720p**).
- Detección de rostro principal (ML Kit Face Detection), obtención de bounding box.
- Recorte ROI + resize (`128×128`), normalización, clasificación con TFLite.
- Etiqueta: nombre + probabilidad (umbral configurable; “Desconocido” si probabilidad < 0.80).
- Suavizado temporal (EMA o ventana de `5–8 frames`).
- Reporte de latencia media/P95 y logging (sin almacenar imágenes).

---

## 3. 📈 Criterios de Éxito

- **Accuracy (hold-out):** ≥ 90% 
- **Macro‑F1:** ≥ 0.90 
- **Dispersión F1:** ≤ 15 pts por clase
- **Latencia:** < 500 ms/frame (media, gama media Android)
- **Consumo:** < 1.5% batería/min; CPU ≤ 70% sostenida
- **Estabilidad:** variaciones de luz, poses ±45°, accesorios (caída ≤ 5 pts)

---

## 4. ⚠️ Supuestos y Riesgos

**Supuestos:**  
- Cara dominante/centrada, distancia 0.5–1.2 m, `15–25 fps` útiles

**Riesgos:**
- Dataset desbalanceado, recortes defectuosos, contraluz, confusión por rasgos similares

**Mitigación:**  
- Ampliar datos difíciles, revisar recortes, ajustar umbral por clase, calibrar augmentación

---

## 5. 📦 Entregables y Trazabilidad

- scripts/*.py (extracción, entrenamiento, evaluación, conversión)
- models/SavedModel_*, .tflite, classes.txt
- reports/metrics_test.json, confusion_matrix.png, training_log.csv
- App Android (android/app)
- README.md/documentación

---

## 6. 🔎 Comprensión de Datos

- **dataset/**: izra/*.jpg, joel/*.jpg, martin/*.jpg
- **Captura:** mínimo 100 imágenes/persona (ideal 200–300); variabilidad en pose, luz, fondo, accesorios
- **Formato:** JPG/PNG, nitidez, sin motion blur
- **EDA (Exploratorio):** conteo por clase/desbalance, balance, diversidad luz/pose, deduplicados
- **Ejemplo `matrix`:**

| Clase  | # Imágenes | % Total | Observaciones      |
|--------|------------|-------- |-------------------|
| izra   | 298        | 33.4 %  | buena variabilidad|
| joel   | 298        | 33.7 %  | faltan exteriores |
| martin | 298        | 32.9 %  | gafas escasas     |

---

## 7. 🛠️ Preparación de Datos

### 7.1 Detección y Recorte
- HaarCascade/Dlib, ROI con margen 10–15%. Resize 128×128 RGB, normalizar `[0,1]`.

### 7.2 Aumentación de Datos (solo train)
| Transformación | Valor     | Nota                         |
|----------------|-----------|------------------------------|
| Rotación       | ±30°      | Evitar >35°                  |
| Shift          | 0.25      | Ancho/alto                   |
| Brillo         | [0.5,1.5] | Contraste realista           |
| Zoom           | 0.3       | Priorizar centrados          |
| Shear          | 0.2       | Perspectiva leve             |
| Flip horizontal| Sí        | No flip vertical             |
| Fill           | nearest   | Rellenar bordes              |

**Regla:** Si `val_accuracy` ↓ y `train_accuracy` ↑, reduce zoom/shear.

### 7.3 Partición
- train/val/test = 70/15/15.
- test totalmente separado (`test_faces128/`).

---

## 8. 🏗️ Modelado

### 8.1 Arquitectura Baseline CNN (ejemplo)
```python
model = keras.Sequential([
    layers.Input(shape=(128,128,3)),
    layers.Conv2D(32, ...),
    layers.BatchNormalization(),
    ... # Ver informe detallado o script
    layers.Dense(num_classes, activation='softmax')
])
```
### 8.2 Hiperparámetros
- lr=5e-5, batch=32, epochs=40–60...
- Optimizer: Adam, Loss: categorical_crossentropy
- Callbacks: ModelCheckpoint, EarlyStopping...

### 8.3 Transfer Learning (opcional)
- MobileNetV2, input 160–192 px, width multiplier 0.75–1.0; excelente para móvil.

### 8.4 Etiquetas y Consistencia
- Generar `classes.txt`, asegurar orden de carpetas = salida del generador

### 8.5 Búsqueda de Hiperparámetros (ejemplo de grid)
| Parámetro   | Valores         |
|-------------|----------------|
| LR          | 1e-4, 7e-5, 5e-5|
| Dropout     | 0.35/0.40/0.50 |
| Filtros     | (32,64,128)/(48,96,192) |
| Img Size    | 128 / 160      |
| Aug         | zoom/shear     |

---

## 9. 📊 Evaluación

### 9.1 Métricas (test)
- Accuracy global, Precision/Recall/F1 por clase, matriz de confusión

### 9.2 Protocolo
- Congelar mejor checkpoint, evaluar en `test_faces128/` sin augmentación

### 9.3 Ejemplo de metrics_test.json
```json
{
  "accuracy_test": 0.93,
  "macro_f1": 0.923,
  "f1_per_class": {"izra": 0.94, "joel": 0.91, "martin": 0.92},
  "support": {"izra": 297, "joel": 302, "martin": 296}
}
```

### 9.4 Criterios de aceptación
- Accuracy ≥ 90%, Macro‑F1 ≥ 0.90, dispersión F1 ≤ 15 pts
- Si F1 < 80% alguna clase: mejorar datos o pipeline

---

## 10. 🚦 Despliegue

### 10.1 Conversión a TFLite (Ejemplo)
```python
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_PATH)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open(TFLITE_PATH, "wb") as f: f.write(tflite_model)
```

### 10.2 Flujo Android (Kotlin)
1. CameraX: captura y preprocesa frame
2. ML Kit: bounding box del rostro principal
3. Procesamiento: resize+normalize a 128×128
4. Inferencia: TFLite Interpreter con multi-thread/NNAPI
5. Postpro: etiquetado y umbral “Desconocido”
6. Suavizado temporal (EMA/ventana)
7. Output: Etiqueta + Probabilidad en pantalla

**Tensor Details:**
- Input: 1×128×128×3 float32 (o int8/float16)
- Output: 1×N_CLASSES (probabilidades)

### 10.3 Pruebas de despliegue
- Latencia real media/P95, resistencia a condiciones variables
- Robustez con y sin suavizado temporal

### 10.4 Telemetría (sin imágenes; opt-in)
- Logging de inferencias “Desconocido”
- Promedio de top-2 predicciones (sin guardar frames)

---

## 11. 🗂️ Detección de rostros: Consideraciones finales

- Mejorar dataset incrementando representatividad y calidad
- Auditar pipeline regularmente para mitigar sesgos y errores sistemáticos

---

### 📌 Referencias y Créditos

- Implementación bajo el marco CRISP-ML(Q), MobileNetV2, TensorFlow Lite, CameraX.
- Contacto: *Equipo FaceID en el Aula (UAC, Cusco, 2025)*

---

*Esta infografía resume el ciclo completo de desarrollo y despliegue de IA biométrica académica en Android, resaltando criterios, riesgos y protocolos para lograr robustez, seguridad y explicabilidad.*


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
