from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("models/skin_disease_model.h5")
class_names = np.load("class_names.npy", allow_pickle=True)

# MUST MATCH TRAINING SIZE
IMG_SIZE = 128

# =========================
# DISEASE LABELS
# =========================
disease_map = {
    "akiec": "Actinic Keratoses (Pre-cancerous lesion)",
    "bcc": "Basal Cell Carcinoma (Skin Cancer)",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma (Dangerous Skin Cancer)",
    "nv": "Melanocytic Nevus (Common Mole)",
    "vasc": "Vascular Lesion"
}

# =========================
# IMAGE PREPROCESSING
# =========================
def preprocess_image(file):

    img = Image.open(file).convert("RGB")

    img = img.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(img).astype("float32")

    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    return img

# =========================
# HOME PAGE
# =========================
@app.route("/")
def home():
    return render_template("index.html")

# =========================
# PREDICTION
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return render_template(
            "index.html",
            prediction="No Image Uploaded"
        )

    file = request.files["image"]

    if file.filename == "":
        return render_template(
            "index.html",
            prediction="No Image Selected"
        )

    try:

        img = preprocess_image(file)

        print("Model Shape:", model.input_shape)
        print("Image Shape:", img.shape)

        pred = model.predict(img, verbose=0)[0]

        pred = pred / np.sum(pred)

        top_indices = pred.argsort()[-4:][::-1]

        results = []

        for i in top_indices:

            code = class_names[i]

            results.append({
                "name": disease_map.get(code, code),
                "score": round(float(pred[i]) * 100, 2)
            })

        best_code = class_names[top_indices[0]]

        prediction = disease_map.get(best_code, best_code)

        confidence = results[0]["score"]

        if confidence >= 80:
            status = "High Confidence"
        elif confidence >= 50:
            status = "Moderate Confidence"
        else:
            status = "Low Confidence"

        return render_template(
            "index.html",
            prediction=prediction,
            confidence=confidence,
            severity=status,
            results=results
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction=f"Error: {str(e)}"
        )

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)