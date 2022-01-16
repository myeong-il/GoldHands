 # -*- coding: utf-8 -*-
import flask
from flask import Flask,render_template,url_for,request
import pickle
import base64
import numpy as np
import cv2
import benchmark_test
app = Flask(__name__)


init_Base64 = 21;


@app.route("/")
def index():
    return render_template('canvas.html')

@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        final_pred = None
        gender = request.form['sex']
        glasses = request.form['glass']
        hair_color = request.form['hair']
        skin_color = request.form['skin']
        file_name = gender + glasses + hair_color + skin_color
        draw = request.form['url']
        draw = draw[init_Base64:]
                    #Decoding
        draw_decoded = base64.b64decode(draw)
        image = np.asarray(bytearray(draw_decoded), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

        cv2.imwrite('static/uploads/0.jpg', image)
        cv2.imwrite('static/uploads/1.jpg', image)

        
        benchmark_test.mainfun(file_name)
        
        return render_template('result.html', image_file = 'predict/tmp.jpg')

if __name__ == '__main__':
    app.run(debug=True)