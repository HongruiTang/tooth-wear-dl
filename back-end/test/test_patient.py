import requests
import unittest
import base64
import json


class TestPatient(unittest.TestCase):

    def test_patient_add(self):
        with open('JawScan_1.ply', 'rb') as f:
            scan = f.read()

        payload = {
            "name": "Test Patient",
            "age": "30",
            "occupation": "Programmer",
            "medicalHistory": "Medical History",
            "painComplaint": "Pain Complaint",
            "financialResources": "Financial Resources",
            "brushingMethod": "Brushing Method",
            "brushingFrequency": "Brushing Frequency",
            "brushingTiming": "Brushing Timing",
            "alcoholIntake": "Alcohol Intake",
            "stressLevel": "Stress Level",
            "sleepApnoea": "Sleep Apnoea",
            "snoringHabit": "Snoring Habit",
            "exercise": "Exercise",
            "drugUse": "Drug Use",
            "upperScan": base64.b64encode(scan).decode('utf-8'),
            "lowerScan": base64.b64encode(scan).decode('utf-8'),
            "sextantScan": base64.b64encode(scan).decode('utf-8')
        }
        json_data = json.dumps(payload)

        response = requests.post('http://20.127.200.67:8080/patient/add', data={'json': json_data})
        data = response.json()['result']
        self.assertEqual(data, '')

    def test_delete_patient(self):
        patient_num_before = requests.get('http://20.127.200.67:8080/patient/number').json()['num']

        with open('back-end/test/JawScan_1.ply', 'rb') as f:
            scan = f.read()

        payload = {
            "name": "Test Patient",
            "age": "30",
            "occupation": "Programmer",
            "medicalHistory": "Medical History",
            "painComplaint": "Pain Complaint",
            "financialResources": "Financial Resources",
            "brushingMethod": "Brushing Method",
            "brushingFrequency": "Brushing Frequency",
            "brushingTiming": "Brushing Timing",
            "alcoholIntake": "Alcohol Intake",
            "stressLevel": "Stress Level",
            "sleepApnoea": "Sleep Apnoea",
            "snoringHabit": "Snoring Habit",
            "exercise": "Exercise",
            "drugUse": "Drug Use",
            "upperScan": base64.b64encode(scan).decode('utf-8'),
            "lowerScan": base64.b64encode(scan).decode('utf-8'),
            "sextantScan": base64.b64encode(scan).decode('utf-8')
        }
        json_data = json.dumps(payload)

        response = requests.post('http://20.127.200.67:8080/patient/add', data={'json': json_data})

        patients = requests.get('http://20.127.200.67:8080/patient/all')
        patient_list = patients.json()['data']
        id = []
        for patient in patient_list:
            id.append(patient[0])

        data = {"id": id[0]}
        response = requests.post('http://20.127.200.67:8080/patient/delete', json=data)

        patient_num_after = requests.get('http://20.127.200.67:8080/patient/number').json()['num']
    
        self.assertEqual(patient_num_after, patient_num_before)

        
      
