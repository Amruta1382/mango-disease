from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__)

# Load trained model
model = load_model("mango_model.h5")

classes = [
    "Anthracnose",
    "Bacterial Canker",
    "Healthy",
    "Powdery Mildew"
]

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# MongoDB Connection
client = MongoClient("mongodb+srv://ammupatil456_db_user:mango1234@cluster0.ilohbix.mongodb.net/?appName=Cluster0")

db = client["mango_disease_db"]
collection = db["predictions"]


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
    img = img / 255
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    index = np.argmax(prediction)
    result = classes[index]
    confidence = float(np.max(prediction))

    # Save prediction in MongoDB
    data = {
        "image_name": file.filename,
        "prediction": result,
        "confidence": round(confidence * 100, 2),
        "date": datetime.now()
    }

    collection.insert_one(data)

    return render_template(
        "result.html",
        prediction=result,
        confidence=round(confidence * 100, 2),
        img=file.filename
    )


if __name__ == "__main__":
    app.run(debug=True)