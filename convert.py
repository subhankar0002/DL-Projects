import tensorflow as tf

# h5 model load
model = tf.keras.models.load_model("dog_cat_model.h5")

# convert to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # quantization
tflite_model = converter.convert()

# save file
with open("dog_cat_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Converted model saved as dog_cat_model.tflite")
