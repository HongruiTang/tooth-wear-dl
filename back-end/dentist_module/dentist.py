from flask import Flask, Blueprint, request
import sqlite3
from sqlite3 import Error
import base64

dentist_bp = Blueprint('dentist', __name__)

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

def encrypt(originalPassword):
    encrypted = base64.b64encode(originalPassword.encode("utf-8"))
    return encrypted

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

@dentist_bp.route('/signin', methods=['POST'])
def dentist_signin():
    data = request.json
    usernameSignIn = data['username']
    passwordSignIn = data['password']

    main(dbFolder)
    conn = create_connection(dbFolder)

    toExecute = "SELECT DENTIST_PASSWORD FROM Dentists WHERE DENTIST_USERNAME = :username"
    crsr = conn.cursor()
    crsr.execute(toExecute, {"username": usernameSignIn})

    result = {}
    try:
        passwordTuple = crsr.fetchall()[0]

        toCheckPassword = []
        for i in passwordTuple:
            toCheckPassword.append(i)

        encryptedPassword = encrypt(passwordSignIn)
        if encryptedPassword == toCheckPassword[0]:
            result = {'result': 'success'}
        else:
            result = {'result': 'fail'}
    except:
        result = {'result': 'invalid'}
    
    return result

@dentist_bp.route('/signup', methods=['POST'])
def dentist_signup():
    data = request.json
    username = data['username']
    password = data['password']
    confirm_password = data['confirm_password']
    encrypted = data['encrypted']

    main(dbFolder)
    conn = create_connection(dbFolder)

    result = {}
    if password == confirm_password:
        if not conn.cursor().execute("INSERT INTO Dentists (DENTIST_USERNAME, DENTIST_PASSWORD) VALUES (?,?)", (username, encrypted)):
            result = {'result': 'fail'}
        else:
            conn.commit()
            crsr = conn.cursor()
            crsr.execute("SELECT * FROM Dentists")
            result = {'result': 'success'}
    else:
        result = {'result': 'invalid'}
    return result
             


