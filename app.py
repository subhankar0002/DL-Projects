from flask import Flask, render_template, request
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Use TFLite model instead of h5
MODEL_PATH = "dog_cat_model.tflite"

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input & output details for prediction
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

app = Flask(__name__)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    if prediction > 0.5:
        return "Dog"
    else:
        return "Cat"

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join("static", file.filename)
            file.save(filepath)
            prediction = predict_image(filepath)
            return render_template("index.html", prediction=prediction, img_path=filepath)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
