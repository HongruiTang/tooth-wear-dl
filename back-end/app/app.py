from flask import Flask, request
from inference_module.inference import inference_bp
from dentist_module.dentist import dentist_bp
from patient_module.patient import patient_bp
import os


# Create an app object using the Flask class
app = Flask(__name__)

app.register_blueprint(inference_bp, url_prefix='/inference')
app.register_blueprint(dentist_bp, url_prefix='/dentist')
app.register_blueprint(patient_bp, url_prefix='/patient')


if __name__ == "__main__":
    # Define port so we can map container port to localhost
    # port = int(os.environ.get('PORT', 5000)) 
    # app.run(host='0.0.0.0', port=port) 
    app.run(host="0.0.0.0", port=8080)
    # app.run()
