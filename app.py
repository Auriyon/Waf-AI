
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load your model once
model = tf.keras.models.load_model('model.h5')

@app.route('/')
def index():
    return render_template('index.html')  # Optional HTML form

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = np.array(data['input']).reshape(1, -1)  # adjust shape for your model
    prediction = model.predict(input_data)
    return jsonify({'prediction': prediction.tolist()})
