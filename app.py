from flask import Flask, render_template, request, jsonify
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import re
from string import punctuation
from tensorflow import keras
from preprocess import Preprocess

@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(punctuation),
                                  '')
model = tf.keras.models.load_model('/Users/Viktoryia/Documents/t/checkpoints_171840')

app = Flask(__name__, template_folder='template')


@app.route('/', methods=['GET'])
def say_hello():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        text = request.form['message']
        csv_preprocess = Preprocess(text)
        csv_data = csv_preprocess.preprocess()
        text_data = pd.DataFrame({'text': [text]}).text
        class_, rating = model.predict([csv_data, text_data])
        rating = np.argmax(rating, axis=-1)[0]

        return render_template('result.html', prediction=class_, rating=rating)


if __name__ == '__main__':
    app.run(debug=True)
