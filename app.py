from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

model = load_model("mango_model.h5")

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
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))