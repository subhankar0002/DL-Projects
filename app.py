from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown

MODEL_PATH = "dog_cat_model.h5"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    url = "https://drive.google.com/file/d/1tcHvT_VwWQr74zMxPiVU2iH9AeF27ieo/view?usp=drive_link"
    gdown.download(url, MODEL_PATH, quiet=False)



app = Flask(__name__)

# Load the trained model
# model = load_model("dog_cat_model.h5")
model = load_model(MODEL_PATH)
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))  # তোমার মডেলের input size
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]  # ধরছি binary classification
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
