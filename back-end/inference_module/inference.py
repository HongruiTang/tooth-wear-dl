from flask import Flask, Blueprint, request
from plyfile import PlyData
import numpy as np
from main import get_prediction

inference_bp = Blueprint('inference', __name__)

@inference_bp.route('/predict', methods=['POST'])
def predict():
    result = {}
    if request.method == 'POST':
        if 'file' not in request.files:
            result =  {'result': 'No file part'}
        file = request.files['file']
        if file == '':
            result =  {'result': 'No file selected for uploading'}
        if file:
            # filename = data['filename']
            # Use werkzeug method to secure filename
            # filename = secure_filename(file)   
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            
            plydata = PlyData.read(file)
            label = get_prediction(plydata)
            # full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            result =  {'result': 'Your tooth wear grade is: {}'.format(label)}
    return result