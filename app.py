from flask import Flask, render_template, request
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__)

# -------------------------------
# Load TFLite model
# -------------------------------
TFLITE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "mango_model.tflite")
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

classes = [
    "Anthracnose",
    "Bacterial Canker",
    "Healthy",
    "Powdery Mildew"
]

# -------------------------------
# Upload folder configuration
# -------------------------------
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------------------
# MongoDB Connection
# -------------------------------
client = MongoClient(
    "mongodb+srv://ammupatil456_db_user:mango1234@cluster0.ilohbix.mongodb.net/?appName=Cluster0"
)
db = client["mango_disease_db"]
collection = db["predictions"]

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    # Save uploaded file
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Load and preprocess image
    img = image.load_img(filepath, target_size=(224, 224))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)  # TFLite expects float32

    # Run TFLite model
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

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