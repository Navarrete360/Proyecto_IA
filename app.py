from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('modelo_perros_gatos.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img_file = request.files['file']
        if img_file:
            img_path = os.path.join('static', img_file.filename)
            img_file.save(img_path)

            # Cargar y preprocesar
            img = image.load_img(img_path, target_size=(128, 128))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_array)[0][0]
            label = 'Perro' if pred >= 0.5 else 'Gato'

            return render_template('index.html', label=label, img_path=img_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
