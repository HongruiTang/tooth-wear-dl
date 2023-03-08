import numpy as np
from flask import Flask, Blueprint, request
from inference_module.inference import inference_bp
from dentist_module.dentist import dentist_bp
from werkzeug.utils import secure_filename
import os


UPLOAD_FOLDER = 'teeth/'

# Create an app object using the Flask class
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.register_blueprint(inference_bp, url_prefix='/inference')
app.register_blueprint(dentist_bp, url_prefix='/dentist')


if __name__ == "__main__":
    # Define port so we can map container port to localhost
    # port = int(os.environ.get('PORT', 5000)) 
    # app.run(host='0.0.0.0', port=port) 
    # app.run(host="0.0.0.0", port=8080)
    app.run()
