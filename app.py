from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import keras
import requests

MODEL_PATH = "dog_cat_model.h5"

if not os.path.exists(MODEL_PATH):
    url = "https://huggingface.co/Subhankar002/dog-vs-cat-classifier/resolve/main/dog_cat_model.h5"
    r = requests.get(url)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
model = keras.models.load_model(MODEL_PATH)


app = Flask(__name__)

# Load the trained model
# model = load_model("dog_cat_model.h5")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256)) 
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]  
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
