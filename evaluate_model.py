
import tensorflow as tf
from sklearn.metrics import classification_report
import numpy as np

# Assuming validation data setup is similar to training
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_height, img_width = 64, 64
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_data = datagen.flow_from_directory(
    'dataset',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

model = tf.keras.models.load_model('model/recyclenet_model.h5')
preds = model.predict(val_data)
y_pred = np.argmax(preds, axis=1)
y_true = val_data.classes

report = classification_report(y_true, y_pred, target_names=val_data.class_indices.keys())
print(report)
