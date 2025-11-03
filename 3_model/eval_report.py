import tensorflow as tf, numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np

IMG=224
test = image_dataset_from_directory("data/test", image_size=(IMG,IMG), batch_size=32, shuffle=False)
class_names = test.class_names
model = tf.keras.models.load_model("models/faceid_best.h5")
y_true=[]; y_pred=[]

for x,y in test:
    p = model.predict(x, verbose=0)
    y_true.extend(y.numpy().tolist())
    y_pred.extend(np.argmax(p, axis=1).tolist())

print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
cm = confusion_matrix(y_true, y_pred)
np.savetxt("models/confusion_matrix.csv", cm, fmt="%d", delimiter=",")
