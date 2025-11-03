import tensorflow as tf

model = tf.keras.models.load_model("models/faceid_best.h5")

# Float16
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.target_spec.supported_types = [tf.float16]
tfl = conv.convert()
open("models/faceid_best_float16.tflite","wb").write(tfl)

# INT8 dynamic range
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
tfl_int8 = conv.convert()
open("models/faceid_best_int8.tflite","wb").write(tfl_int8)

print("TFLite exportado")
