# FaceID en el Aula: Construcción y Despliegue de un Sistema de Reconocimiento Facial con Deep Learning

**Resumen.** Presentamos un pipeline CRISP‑ML para reconocimiento facial de 5 integrantes con MobileNetV2, entrenado con transfer learning, y desplegado en Android con TensorFlow Lite. Se reportan métricas por clase, matriz de confusión y latencia medida en dispositivo real.

## 1. Introducción
El reconocimiento facial basado en CNN ofrece buen compromiso entre precisión y costo. Transfer learning reduce datos y tiempo de cómputo.

## 2. Metodología CRISP‑ML
### 2.1 Comprensión del Negocio
Meta: accuracy test ≥90% y latencia <500 ms. Riesgos: iluminación, oclusiones.
### 2.2 Comprensión de los Datos
≥100 imágenes por persona, variando pose, iluminación, expresión; EDA con balance de clases.
### 2.3 Preparación de Datos
Detección y recorte facial, normalización a 224×224, augment leve y split 70/15/15.
### 2.4 Modelado
MobileNetV2 (ImageNet), capas congeladas y fine‑tuning en últimas capas. Optimización Adam.
### 2.5 Evaluación
Accuracy, Precision, Recall, F1 por clase y matriz de confusión en test.
### 2.6 Despliegue
Conversión a TFLite (float16/INT8), CameraX para stream e inferencia en tiempo real.

## 3. Resultados y Discusión
Incluir: tabla de métricas por clase, curva loss/acc y matriz de confusión exportada por `eval_report.py`.

## 4. Despliegue Móvil
Android minSdk 24. Entrada 224×224. 2–4 hilos según dispositivo. Recorte facial previo para robustez.

## 5. Conclusiones y Trabajo Futuro
Más datos, face alignment, distillation y regularización adicional. Evaluar ArcFace/triplet loss si hay personas muy parecidas.
