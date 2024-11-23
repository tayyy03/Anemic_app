from flask import Flask, request, jsonify
from PIL import Image
import io
import tensorflow as tf

app = Flask(__name__)

# Muat model ML yang sudah Anda latih
model = tf.keras.models.load_model("D:/upwork/anemia/cnn_model.h5")  # Ganti dengan path model Anda
labels = ["No Anemia", "Anemia"]  # Label klasifikasi

@app.route("/")
def home():
    return "Anemia Detection API is running!"

@app.route("/detect-anemia", methods=["POST"])
def detect_anemia():
    if "image" not in request.files:
        return jsonify({"error": "No image file found"}), 400

    file = request.files["image"]
    try:
        # Buka dan preproses gambar
        image = Image.open(file.stream).convert("RGB").resize((224, 224))
        image_array = tf.keras.utils.img_to_array(image) / 255.0  # Normalisasi
        image_array = tf.expand_dims(image_array, axis=0)  # Tambahkan batch dimensi

        # Inferensi menggunakan model
        prediction = model.predict(image_array)
        result = labels[int(tf.argmax(prediction, axis=1))]

        return jsonify({"result": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
