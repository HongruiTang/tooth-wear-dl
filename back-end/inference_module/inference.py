from flask import Blueprint, request
from plyfile import PlyData
from inference_module.main import get_prediction
import sqlite3
from sqlite3 import Error

inference_bp = Blueprint('inference', __name__)

dbFolder = "../ToothWear.db"

def create_connection(db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
        except Error as e:
            print(e)
        return conn

@inference_bp.route('/predict', methods=['POST'])
def predict():
    result = {}
    if request.method == 'POST':
        patientID = request.files['id']
        conn = create_connection(dbFolder)
        toExecute = "SELECT PATIENT_SEXTANT_SCAN FROM Patients WHERE PATIENT_ID = :id"
        crsr = conn.cursor()
        crsr.execute(toExecute, {"id": patientID})
        sextant = crsr.fetchall()[0]
        
        plydata = PlyData.read(sextant)
        label = get_prediction(plydata)
        result =  {'result': 'Your tooth wear grade is: {}'.format(label)}
    return result