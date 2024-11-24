from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import cv2  # Pastikan OpenCV terinstal
# Import model Anda di sini
# from your_model import load_model, predict

app = Flask(__name__)
CORS(app)  # Mengizinkan CORS untuk semua domain (Anda dapat membatasi ini sesuai kebutuhan)

# Mengatur direktori untuk menyimpan gambar yang diunggah
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Simpan file yang diunggah
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Proses gambar dan lakukan prediksi
    result = classify_image(file_path)

    # Hapus file setelah pemrosesan (opsional)
    os.remove(file_path)

    return jsonify({'result': result})

def classify_image(image_path):
    # Muat model Anda di sini
    # model = load_model()

    # Baca gambar
    image = cv2.imread(image_path)

    # Lakukan pra-pemrosesan yang diperlukan untuk model Anda
    # Misalnya, ubah ukuran gambar, normalisasi, dll.
    # image = preprocess_image(image)

    # Lakukan prediksi
    # prediction = model.predict(image)

    # Contoh hasil prediksi (ganti dengan logika Anda)
    # Jika model Anda mengembalikan probabilitas, Anda dapat menggunakan ambang batas untuk menentukan hasil
    # return 'Anemia' if prediction[0] > 0.5 else 'No Anemia'
    
    # Untuk contoh, kita kembalikan hasil acak
    return 'Anemia'  # Ganti dengan logika deteksi yang sebenarnya

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
