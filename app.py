from flask import Flask, render_template, request
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

MODEL_PATH = "mango_model.h5"

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000:
    print("Downloading model...")
    url = os.environ["MODEL_URL"]
    r = requests.get(url, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Model downloaded successfully.")

model = load_model(MODEL_PATH)


classes = [
    "Anthracnose",
    "Bacterial Canker",
    "Healthy",
    "Powdery Mildew"
]

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224,224))
    img = image.img_to_array(img)
    img = img/255
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    index = np.argmax(prediction)
    result = classes[index]
    confidence = float(np.max(prediction))

    return render_template(
        "result.html",
        prediction=result,
        confidence=round(confidence*100,2),
        img=file.filename
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)