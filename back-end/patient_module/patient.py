from flask import Flask, Blueprint, request

patient_bp = Blueprint('patient', __name__)

@patient_bp.route('/patient', methods=['POST'])
def patient():
    pass