from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import requests

# Configuración del modelo
MODEL_URL = "https://drive.google.com/uc?id=10X740-nRzbjDpCM5odprGas6KgQFV2ME"
MODEL_PATH = "modelo_perros_gatos.h5"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Descargando modelo desde Google Drive...")
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Modelo descargado.")

# Descarga el modelo si no existe
download_model()

# Carga del modelo
model = load_model(MODEL_PATH)

# App Flask
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img_file = request.files['file']
        if img_file:
            img_path = os.path.join('static', img_file.filename)
            img_file.save(img_path)

            # Cargar y preprocesar la imagen
            img = image.load_img(img_path, target_size=(128, 128))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predicción
            pred = model.predict(img_array)[0][0]
            label = 'Perro' if pred >= 0.5 else 'Gato'

            return render_template('index.html', label=label, img_path=img_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
