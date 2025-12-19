from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Chạy model
model = tf.keras.models.load_model("keras_model.h5")

# Chạy nhãn
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.asarray(image).astype(np.float32)
    image = (image / 127.5) - 1
    return np.expand_dims(image, axis=0)

# Dự đoán
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    image = Image.open(file).convert("RGB")

    input_data = preprocess_image(image)
    predictions = model.predict(input_data)[0]

    index = np.argmax(predictions)
    result = {
        "label": labels[index],
        "confidence": float(predictions[index])
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
