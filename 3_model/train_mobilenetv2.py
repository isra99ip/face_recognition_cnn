import tensorflow as tf, os
from tensorflow.keras import layers, models

IMG = 224; BATCH=32; EPOCHS=15
train = tf.keras.preprocessing.image_dataset_from_directory("data/train", image_size=(IMG,IMG), batch_size=BATCH)
val   = tf.keras.preprocessing.image_dataset_from_directory("data/val",   image_size=(IMG,IMG), batch_size=BATCH)

AUTOTUNE=tf.data.AUTOTUNE
train = train.shuffle(1000).prefetch(AUTOTUNE)
val   = val.prefetch(AUTOTUNE)

data_augment = tf.keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.1),
  layers.RandomZoom(0.1),
])

base = tf.keras.applications.MobileNetV2(input_shape=(IMG,IMG,3), include_top=False, weights="imagenet")
base.trainable = False

inputs = layers.Input(shape=(IMG,IMG,3))
x = data_augment(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(train.class_names), activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

os.makedirs("models", exist_ok=True)
ckpt="models/faceid_best.h5"
callbacks=[tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_accuracy", save_best_only=True, mode="max"),
           tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
model.fit(train, validation_data=val, epochs=EPOCHS, callbacks=callbacks)

# Fine-tuning
base.trainable = True
for layer in base.layers[:-30]: 
    layer.trainable=False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train, validation_data=val, epochs=10, callbacks=callbacks)
print("saved:", ckpt)

with open("3_model/labels.txt","w") as f:
    f.write("\n".join(train.class_names))
