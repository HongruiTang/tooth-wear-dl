import requests
import unittest


class TestInference(unittest.TestCase):

    def test_predict_success(self):
        response = requests.post('http://20.127.200.67:8080//inference/predict', json={'id': 1})
        data = response.json()['result']
        self.assertEqual(data, 'Your tooth wear grade is: 2')

    def test_predict_fail(self):
        response = requests.post('http://20.127.200.67:8080//inference/predict', json={'id': ''})
        data = response.json()['result']
        self.assertEqual(data, 'fail')

    