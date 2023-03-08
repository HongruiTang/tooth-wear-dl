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

def encrypt(originalPassword):
    encrypted = base64.b64encode(originalPassword.encode("utf-8"))
    return encrypted

@dentist_bp.route('/dentist/signin', methods=['POST'])
def dentist_signin():
    data = request.json
    usernameSignIn = data['username']
    passwordSignIn = data['password']

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

