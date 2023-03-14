from flask import Blueprint, request
import sqlite3
import base64
import json
from sqlite3 import Error

patient_bp = Blueprint('patient', __name__)

dbFolder = "../ToothWear.db"


def create_connection(db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
        except Error as e:
            print(e)
        return conn

def create_table(conn, create_table_sql):
        try:
            c = conn.cursor()
            c.execute(create_table_sql)
        except Error as e:
            print(e)

def main(dbFolder):
        create_patient_table = """ CREATE TABLE IF NOT EXISTS Patients (
                                    PATIENT_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                                    PATIENT_NAME TEXT,
                                    PATIENT_AGE TEXT,
                                    PATIENT_OCCUPATION TEXT,
                                    PATIENT_MEDICAL_HISTORY TEXT,
                                    PATIENT_PAIN_COMPLAINT TEXT,
                                    PATIENT_FINANCIAL_RESOURCES TEXT,
                                    PATIENT_BRUSHING_METHOD TEXT,
                                    PATIENT_BRUSHING_FREQUENCY TEXT,
                                    PATIENT_BRUSHING_TIMING TEXT,
                                    PATIENT_ALCOHOL_INTAKE TEXT,
                                    PATIENT_STRESS_LEVEL TEXT,
                                    PATIENT_SLEEP_APNOEA TEXT,
                                    PATIENT_SNORING_HABIT TEXT,
                                    PATIENT_EXERCISE TEXT,
                                    PATIENT_DRUG_USE TEXT,
                                    PATIENT_UPPER_JAW_SCAN BLOB,
                                    PATIENT_LOWER_JAW_SCAN BLOB,
                                    PATIENT_SEXTANT_SCAN BLOB
                                );
                            """
        create_dentist_table = """ CREATE TABLE IF NOT EXISTS Dentists (
                                    DENTIST_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                                    DENTIST_USERNAME TEXT,
                                    DENTIST_PASSWORD TEXT
                                );
                            """
        conn = create_connection(dbFolder)

        if conn is not None:
            create_table(conn, create_patient_table)
            create_table(conn, create_dentist_table)
        else:
            print("Error! Cannot create a patient database connection")

@patient_bp.route('/add', methods=['POST'])
def add_patient():
    conn = create_connection(dbFolder)
    data = request.json

    name = data['name']
    age = data['age']
    occupation = data['occupation']
    medicalHistory = data['medicalHistory']
    painComplaint = data['painComplaint']
    financialResources = data['financialResources']
    brushingMethod = data['brushingMethod']
    brushingFrequency = data['brushingFrequency']
    brushingTiming = data['brushingTiming']
    alocholIntake = data['alcoholIntake']
    stressLevel = data['stressLevel']
    sleepApnoea = data['sleepApnoea']
    snoringHabit = data['snoringHabit']
    exercise = data['exercise']
    drugUse = data['drugUse']
    # upperScan = data['upperScan']['file']
    # lowerScan = data['lowerScan']['file']
    # sextantScan = data['sextantScan']['file']
    upperScan = request.files['upperScan']
    lowerScan = request.files['lowerScan']
    sextantScan = request.files['sextantScan']

    result = {'result': ''}
    crsr = conn.cursor()
    if not crsr.execute('INSERT INTO Patients (PATIENT_NAME, PATIENT_AGE, PATIENT_OCCUPATION, PATIENT_MEDICAL_HISTORY, PATIENT_PAIN_COMPLAINT, PATIENT_FINANCIAL_RESOURCES, PATIENT_BRUSHING_METHOD, PATIENT_BRUSHING_FREQUENCY,PATIENT_BRUSHING_TIMING, PATIENT_ALCOHOL_INTAKE, PATIENT_STRESS_LEVEL, PATIENT_SLEEP_APNOEA, PATIENT_SNORING_HABIT, PATIENT_EXERCISE, PATIENT_DRUG_USE, PATIENT_UPPER_JAW_SCAN, PATIENT_LOWER_JAW_SCAN, PATIENT_SEXTANT_SCAN) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (name, age, occupation, medicalHistory, painComplaint, financialResources, brushingMethod, brushingFrequency, brushingTiming, alocholIntake, stressLevel, sleepApnoea, snoringHabit, exercise, drugUse, upperScan, lowerScan, sextantScan)):
        result = {'result': 'fail'}
    else:
        conn.commit()
    return result

@patient_bp.route('/delete', methods=['POST'])
def delete_patient():
    conn = create_connection(dbFolder)
    data = request.json
    patientID = data['id']
    toExecute = "DELETE FROM Patients WHERE PATIENT_ID = :id"
    crsr = conn.cursor()
    crsr.execute(toExecute, {"id": patientID})
    conn.commit()
        

@patient_bp.route('/number', methods=['GET'])
def get_patient_number():
    conn = create_connection(dbFolder)
    toExecute = "SELECT MAX(PATIENT_ID) FROM Patients"
    crsr = conn.cursor()
    crsr.execute(toExecute)

    numberOfPatients = crsr.fetchall()[0]
    if numberOfPatients[0] is None:
        return {'num': 0}

    return {'num': numberOfPatients[0]}

@patient_bp.route('/all', methods=['GET'])
def get_all_patients():
    conn = create_connection(dbFolder)
    get_all_patients = """
                        SELECT PATIENT_ID, PATIENT_NAME, PATIENT_AGE, PATIENT_OCCUPATION, PATIENT_MEDICAL_HISTORY, PATIENT_PAIN_COMPLAINT, PATIENT_FINANCIAL_RESOURCES, PATIENT_BRUSHING_METHOD, PATIENT_BRUSHING_FREQUENCY,PATIENT_BRUSHING_TIMING, PATIENT_ALCOHOL_INTAKE, PATIENT_STRESS_LEVEL, PATIENT_SLEEP_APNOEA, PATIENT_SNORING_HABIT, PATIENT_EXERCISE, PATIENT_DRUG_USE FROM Patients;           
                       """
    try:
        crsr = conn.cursor()
        crsr.execute(get_all_patients)
        data = crsr.fetchall()
        return {'data': data}
    except Error as e:
        print(e)

@patient_bp.route('/view', methods=['POST'])
def view():
    data = request.json
    patientID = data['id']
    conn = create_connection(dbFolder)
    toExecute = "SELECT PATIENT_UPPER_JAW_SCAN, PATIENT_LOWER_JAW_SCAN FROM Patients WHERE PATIENT_ID = :id"
    crsr = conn.cursor()
    crsr.execute(toExecute, {"id": patientID})

    upper_file, lower_file = crsr.fetchall()[0]
    return {'upper': upper_file, 'lower': lower_file}
