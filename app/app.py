from flask import Flask, render_template, request
import pandas as pd
import os
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Dictionary mapping class labels to music genres
music_genre = {
    1: 'pop',
    2: 'metal',
    3: 'disco',
    4: 'blues',
    5: 'reggae',
    6: 'classical',
    7: 'rock',
    8: 'hiphop',
    9: 'country',
    10: 'jazz',
}

def extractAudio(filename):
    path = '../Data/features_30_sec.csv'
    dataset = pd.read_csv(path)
    row = dataset.loc[dataset['filename'] == filename]
    return row

def preProcess(data):
    features = data.drop('filename', axis=1)
    X = np.array(features.drop('label', axis=1))
    y_true = np.array(data['label'])
    return X, y_true

def predictGenre(X):
    model = load_model('../CNN/cnn_model6_30_features.h5')
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred)
    predicted_genre = music_genre.get(y_pred)
    return predicted_genre

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message='No file selected')
    if file:
        filename = file.filename
        data_row = extractAudio(filename)
        X, y_true = preProcess(data_row)
        predicted_genre = predictGenre(X)
        return render_template('result.html', filename=filename, actual_genre=y_true, predicted_genre=predicted_genre)

if __name__ == '__main__':
    app.run(debug=True)
